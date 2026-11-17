"""Annotate sequences with everything Quill knows: DNA, RNA and protein.

    python3 quill_annotate.py input.fa [--size nano|small|medium|large] [--out a.json]

Each FASTA record may be genomic DNA, an mRNA/cDNA or a protein. For nucleotide records:
  dna         bits per base under quill-s; Alu hits; written TF sites (>= 12 bits) and
              the strongest JASPAR matrices
  rna         the longest ORF of at least 100 codons splits the record into 5'UTR, CDS
              and 3'UTR, which are read with the written RNA knowledge: uORFs, start
              context, 5'TOP, G-quadruplexes, AU-rich, Pumilio and CPE elements, poly(A)
              signal, RNA-binding-protein sites, microRNA seed matches, codon adaptation
              and 5'UTR folding with ViennaRNA. With no such ORF the record is classed
              by the written Sequence Ontology model instead
For the protein (translated, or given):
  protein     mass, pI, hydropathy, signal peptide and cleavage site, transmembrane
              helices and topology, mitochondrial presequence, GPI anchor, disorder,
              coiled coils, low complexity, short linear motifs, PROSITE patterns,
              the written signatures of domains.py
  function    quill-xl3 Gene Ontology terms in all three namespaces (most specific
              confident terms), and Enzyme Commission numbers through GO
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

STOPS = {'TAA', 'TAG', 'TGA'}
COMP = str.maketrans('ACGTN', 'TGCAN')
MIN_CODONS = 100
MIN_SITE_BITS = 12.0


def read_fasta(path):
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(buf).upper()
                name, buf = (line[1:].split() or ['seq'])[0], []
            elif line:
                buf.append(line)
    if name is not None:
        yield name, ''.join(buf).upper()


def longest_orf(dna):
    """(strand, transcript in that orientation, start, end) of the longest ATG ORF."""
    best = None
    for strand, s in (('+', dna), ('-', dna.translate(COMP)[::-1])):
        for frame in range(3):
            i = frame
            while i + 3 <= len(s):
                if s[i:i + 3] == 'ATG':
                    j = i
                    while j + 3 <= len(s) and s[j:j + 3] not in STOPS:
                        j += 3
                    if best is None or j - i > best[3] - best[2] - 3:
                        best = (strand, s, i, min(j + 3, len(s)))
                    i = j + 3
                else:
                    i += 3
    return best


class Annotator:
    def __init__(self, size='medium', top=8):
        import knowledge as K
        import ontology as O
        import qdomains
        import qgox
        import qnongenic as NG
        import qprot
        import qrna
        import qso
        import qxl
        import qxl3
        import regulatory as R
        self.K, self.O, self.NG, self.qprot, self.qrna, self.qxl = K, O, NG, qprot, qrna, qxl
        self.top = top
        self.meta = qgox.load_terms()
        self.m = qxl3.XL3(qxl.MODELS / f'quill-xl3-{size}.npz')
        self.ecm = qxl.ec_map(None)
        self.rules = qdomains.resolve(self.meta)[0]
        self.qdomains = qdomains
        self.so = qso.SoModel(qso.MODELS / 'quill-k-so.npz')
        self.wnames = list(R.TF_SITES)
        self.wbank = NG.motif_bank([np.log2(np.asarray(R.matrix(s)) / 0.25)
                                    for s in R.TF_SITES.values()])
        self.wmax = np.array([float(np.log2(np.asarray(R.matrix(s)).max(1) / 0.25).sum())
                              for s in R.TF_SITES.values()])
        try:
            J = NG.jaspar()
            self.jnames = [n for n, _, _ in J]
            self.jbank = NG.motif_bank([lo for _, _, lo in J])
            self.jmax = np.array([float(lo.max(1).sum()) for _, _, lo in J])
        except Exception:
            self.jbank = None
        info = {a: (n, d) for a, n, d, _, _, _ in qprot.prosite_patterns()}
        self.prosite_info = info
        try:
            from qbench import Model
            import qanno as A
            import qinfer as I
            self.seqmodel, self.A, self.I = Model('quill-s'), A, I
        except Exception:
            self.seqmodel = None
        children = {}
        for i in range(self.m.T):
            for k in range(self.m.parent_ptr[i], self.m.parent_ptr[i + 1]):
                children.setdefault(int(self.m.parent_idx[k]), []).append(i)
        self.children = children

    def specific(self, P, ns, th=0.3):
        conf = {i for i, t in enumerate(self.m.names)
                if self.meta[t]['ns'] == ns and P[i] >= th and P[i] >= 1.2 * self.m.prior[i]}
        covered = {int(self.m.parent_idx[k]) for i in conf
                   for k in range(self.m.parent_ptr[i], self.m.parent_ptr[i + 1])}
        keep = sorted(conf - covered, key=lambda i: -P[i] / self.m.prior[i])[:self.top]
        return [dict(go=f'GO:{self.m.names[i]}', name=self.meta[self.m.names[i]]['name'],
                     p=round(float(P[i]), 3), prior=round(float(self.m.prior[i]), 4))
                for i in keep]

    def function(self, prot, rna_desc):
        d = self.qprot.describe(prot)
        hits = self.qprot.prosite(prot)
        rf = self.qrna.features(rna_desc) if rna_desc else []
        P = self.m.predict([prot], {'protein': [self.qprot.features(d, ())],
                                    'prosite': [[f'prosite:{h}' for h in hits]],
                                    'rna': [rf], 'dna': [[]]})[0]
        homologues = [dict(symbol=self.m.symbols[j], score=int(sc), relative=round(r, 3))
                      for j, sc, r in self.m.aligner.ranked(prot)[:5]] \
            if self.m.aligner is not None else []
        ec = {}
        for i, t in enumerate(self.m.names):
            if P[i] >= 0.3:
                for e in self.ecm.get(t, ()):
                    if '-' not in e:
                        ec[e] = max(ec.get(e, 0), float(P[i]))
        sig = [dict(signature=r, go=f'GO:{t}', name=self.meta[t]['name'],
                    written_confidence=c) for r, t, c in self.qdomains.scan(prot, self.rules)]
        prot_report = {k: v for k, v in d.items() if k != 'slims'}
        prot_report['motifs'] = {k: v for k, v in d['slims'].items() if v}
        prot_report['prosite'] = [dict(acc=h, name=self.prosite_info[h][0],
                                       description=self.prosite_info[h][1]) for h in hits]
        prot_report['written_signatures'] = sig
        return prot_report, dict(
            homologues=homologues, MF=self.specific(P, 'MF'), BP=self.specific(P, 'BP'),
            CC=self.specific(P, 'CC'),
            EC=[dict(ec=e, p=round(p, 3)) for e, p in
                sorted(ec.items(), key=lambda kv: -kv[1])[:5]])

    def annotate(self, name, seq):
        rec = dict(id=name, length=len(seq))
        nucleotide = set(seq) <= set('ACGTUN')
        if not nucleotide:
            rec['protein'], rec['function'] = self.function(seq.replace('*', ''), None)
            return rec
        dna = seq.replace('U', 'T')
        arr = np.array([{'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(c, 4) for c in dna], np.int8)
        rec['dna'] = dict(gc=round(float(np.mean([c in 'GC' for c in dna if c in 'ACGT'])), 3))
        if self.seqmodel is not None and len(arr) >= 50:
            gcb = self.A.gc_bins(arr, self.A.GC_WIN, *self.A.GC_EDGES)
            lp, n, _ = self.I.forward(arr, gcb, *self.seqmodel.args(), False)
            rec['dna']['bits_per_base'] = round(float(-lp / max(n, 1)), 4)
        rec['dna']['alu_head_hits'] = int(len(self.NG.kmer_hits(arr, [self.O.ALU_HEAD], 16))) \
            if len(arr) > 16 else 0
        w = self.NG.motif_max(arr, self.wbank, len(self.wnames)) / self.wmax
        rec['dna']['written_tf_sites'] = [dict(tf=self.wnames[i], fraction_of_max=round(float(w[i]), 3))
                                          for i in np.argsort(-w) if w[i] >= 0.9 and
                                          self.wmax[i] >= MIN_SITE_BITS][:self.top]
        if self.jbank:
            j = self.NG.motif_max(arr, self.jbank, len(self.jnames)) / self.jmax
            rec['dna']['jaspar'] = [dict(tf=self.jnames[i], fraction_of_max=round(float(j[i]), 3))
                                    for i in np.argsort(-j)[:self.top]]
        orf = longest_orf(dna)
        if orf and (orf[3] - orf[2]) // 3 >= MIN_CODONS:
            strand, s, a, b = orf
            u5, cds, u3 = s[:a], s[a:b], s[b:]
            prot = ''.join(self.K.GENETIC_CODE.get(cds[i:i + 3], 'X')
                           for i in range(0, len(cds) - 2, 3)).rstrip('*')
            rd = self.qrna.describe(u5, cds, u3, [], len(cds), do_fold=True)
            rec['rna'] = dict(orf=dict(strand=strand, start=a, end=b, codons=len(prot)),
                              **{k: v for k, v in rd.items() if k not in ('rbp_3utr', 'mirna_sites')},
                              rbp_sites_3utr={k: v for k, v in rd['rbp_3utr'].items() if v},
                              mirna_sites_3utr={k: v for k, v in rd['mirna_sites'].items() if v})
            rec['protein'], rec['function'] = self.function(prot, rd)
        elif len(arr) <= 6000:
            sc = self.so.score(arr, 1)
            p = np.exp2(sc - sc.max())
            p /= p.sum()
            mfe, z = self.qrna.fold_z(dna[:300])
            rec['rna'] = dict(noncoding=[dict(so=self.so.names[i],
                                              term=self.O.SO_RNA_CLASSES[self.so.names[i]]['so'],
                                              p=round(float(p[i]), 3)) for i in np.argsort(-p)[:3]],
                              fold_mfe=round(mfe, 2), fold_z=round(z, 2))
        return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('fasta')
    ap.add_argument('--size', default='medium', choices=['nano', 'small', 'medium', 'large'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()
    ann = Annotator(a.size, a.top)
    recs = [ann.annotate(n, s) for n, s in read_fasta(a.fasta)]
    txt = json.dumps(recs, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    if a.out:
        Path(a.out).write_text(txt)
        print(f'wrote {a.out}: {len(recs)} records')
    else:
        print(txt)


if __name__ == '__main__':
    main()
