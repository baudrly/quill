# Quill

Three mammalian DNA language models that share one hand-written gene-structure grammar
and differ only in where their weights came from. No model here takes a gradient step.

| Model | Parameters | On disk | Weights from |
| --- | --- | --- | --- |
| `quill-k` | 10,252 | 5.2 KB | `knowledge.py` alone — no genome, no annotation, no counting |
| `quill-s` | 1,406,988 | 0.96 MB | written grammar + statistics counted from 2.65 Gb of human sequence |
| `quill-l` | 67,467,276 | 49.9 MB | the same, with a larger k-mer backbone |

Held out and never counted: human chromosomes 17, 19, 20, 21, 22, and mouse entirely.

Model files and card on [huggingface](https://huggingface.co/baudrly/quill)

## Headline results

- Held-out human: 1.925 / 1.756 / 1.668 bits per base (k / s / l).
- Held-out mouse chr19: 1.922 / 1.904 / 1.898. Counting 2.65 Gb of human DNA buys 0.26
  bits/base on human and 0.02 on mouse.
- BRCA1 noncoding variants (1,125 SNVs, the file the Evo 2 repo ships): the 176
  splice-matrix parameters give AUROC **0.873** counted, **0.790** written, against a
  published 0.87 for Evo 2 7B and 0.875 for CADD.
- Missense variants and every likelihood-based scoring head: at or near chance.

## The two ontology layers

Three more written models sit on top of the sequence grammar, built from two further
declarative files and evaluated on the same held-out chromosomes.

| Model | Parameters | Written from | Held-out result |
| --- | --- | --- | --- |
| `quill-k-so` | 260 | `ontology.py` | SO class of a non-coding transcript, macro AUROC **0.876** (counted twin 0.868) |
| `quill-k-fam` | 24,298 | `function.py` | gene family from coding sequence, macro AUROC **0.986** (counted twin 0.945) |
| `quill-k-go` | 34,684 | `function.py` | GO slim class, macro AUROC 0.662 (counted twin 0.680) |

Written weights beat counted ones on the four rarest families, because counting needs
examples and writing does not. A written 8x4 matrix localises the U12 minor-spliceosome
branch point at AUROC 0.822 among 38,692 held-out introns. No written core-promoter
element reaches 0.60 against a composition-matched control.

## quill-xl: full-size functional knowledge

No size limit, still no gradients. Evaluated on held-out chromosomes 17, 19, 20, 21, 22.

| Part | What it knows | Held-out result |
| --- | --- | --- |
| `quill-xl-go.npz` (13 M params, 44 MB) | 4,175 GO terms: sparse peptide 5-mer dictionary + gene memory + GO graph | MF Fmax 0.615 vs 0.410 naive; 0.548 vs 0.393 on remote genes |
| `domains.py` | 67 written protein signatures, each tied to a GO term by name | 36/42 testable rules significant (Bonferroni), median precision 0.94 |
| `regulatory.py` | 33 written TF sites | median 97% of the JASPAR matrix maximum; best regulatory features on 4/4 tasks |
| `ontology.py` Alu head | written repeat consensus | 119-239 loci/Mb vs ~0 for shuffles (the written L1 string was wrong) |

`quill-xl-go-full.npz` (64 MB) is the release model: it remembers all 19,016 annotated
human genes and is never used for a held-out number.

```sh
python3 quill_annotate.py input.fa --out annotations.json
```

## quill-xl2: DNA, RNA and protein experts, four sizes

| Model | Parameters | Size | GO terms | MF Fmax | EC Fmax | Where |
| --- | --- | --- | --- | --- | --- | --- |
| term frequency | | | | 0.410 | 0.267 | |
| `quill-xl2-nano` | 97 k | 0.28 MB | 2,387 | 0.539 | 0.429 | browser (Gene function tab) |
| `quill-xl2-small` | 307 k | 1.1 MB | 4,175 | 0.580 | 0.473 | anywhere |
| `quill-xl2-medium` | 13.1 M | 45 MB | 4,175 | 0.627 | 0.557 | laptop; the default |
| `quill-xl2-large` | 15.9 M | 56 MB | 6,230 | 0.613 | 0.566 | laptop |

Experts: k-mer dictionary, gene memory, written protein features (`proteome.py`), PROSITE
patterns, written mRNA features with ViennaRNA folding (`transcriptome.py`), written
signatures (`domains.py`). Ontologies: full GO, EC numbers through GO, the generic GO slim,
and the Sequence Ontology classes for non-coding RNA. No gradient descent anywhere.

## quill-xl3 and quill-v

| Model | Parameters | BP | MF | CC | EC |
|---|---|---|---|---|---|
| xl3-nano (browser) | 97,273 | 0.413 | 0.541 | 0.632 | 0.433 |
| xl3-small | 360,636 | 0.425 | 0.580 | 0.636 | 0.474 |
| xl3-medium | 21,201,398 | 0.551 | 0.690 | 0.679 | 0.672 |
| xl3-large | 24,022,344 | 0.554 | 0.704 | 0.679 | 0.694 |

- `qalign.py` Smith-Waterman alignment memory (BLOSUM62, parasail): the gain.
- `qdna.py` DNA-level gene architecture / promoter / neighbourhood expert: small CC gain.
- GO cross-talk and two-hop homology: tested, rejected.
- `qvariant.py` written variant-effect rules: BRCA1 AUROC 0.772 (missense 0.587).
- `qxl3.py build|bench`, `qverify5.py` (205 checks), `MODEL_CARD.md`.
- `quill_annotate.py` now uses quill-xl3 and lists the aligned homologues behind each call.

```sh
python3 quill_annotate.py input.fa --size medium   # DNA, mRNA or protein
```

## Layout

```
knowledge.py    every number the zero-data sequence model is built from, by hand
ontology.py     Sequence Ontology knowledge: 21 matrices, 10 RNA classes, cis elements
function.py     Gene Ontology knowledge: 10 slim classes, 7 families, periods, targeting
qwritten.py     turns knowledge.py into models/quill-k.npz
qstates.py      the 146-state gene grammar and the annotation -> state path map
qinfer.py       forward, Viterbi, sampling, variant scoring kernels (numba)
qanno.py        annotation parsing, GC bins, CpG islands, tandem repeats
qcount.py       k-mer counting over the training chromosomes
qbuild.py       counts -> interpolated, log-quantised parameter tables
qmodel.py       assembles quill-s and quill-l, incl. EM over the mixture weights
qbench.py       all benchmarks -> results/bench.json
qeval_bb.py     backbone-only scaling curve -> results/scaling.json
qfigs.py        figures;  qmakefigs.py is the entry point
qexport.py      exports a model as one JS file for the browser
qparity.py      writes the browser-parity reference
qverify.py      checks every number in the preprint against results/
qevo2.py        head-to-head against Evo 2 (needs a GPU)
qgoparse.py     pulls GO terms and protein translations out of the Ensembl EMBL files
qgo.py          function-conditioned coding model: build, score, classify
qso.py          Sequence Ontology model: build, score, classify
qgobench.py     function benchmarks -> results/gobench.json
qsobench.py     sequence-ontology benchmarks -> results/sobench.json
qontexport.py   exports the three ontology models for the browser
qontexamples.py ships real held-out genes with the page
qontparity.py   writes the ontology browser-parity reference
qverify2.py     checks every ontology number in the preprint
domains.py      67 written protein signatures -> GO terms
regulatory.py   33 written transcription-factor sites
qgox.py         quill-xl gene side: GO graph, dictionary, gene memory
qgoxbench.py    held-out whole-GO Fmax with baselines -> results/goxbench.json
qdomains.py     tests each written signature -> results/domains.json
qnongenic.py    TF sites vs JASPAR, regulatory probe, repeats -> results/nongenic.json
quill_annotate.py  one command over a FASTA, every layer
qsigexport.py   ships the written signatures to the browser
qverify3.py     checks every quill-xl number in the preprint
proteome.py     written protein knowledge: scales, sorting signals, topology, motifs
transcriptome.py written RNA knowledge: UTR elements, RBP sites, miRNA seeds, folding
qprot.py        describes a protein; PROSITE patterns -> data/gox/prot.pkl
qrna.py         assembles and describes mRNAs; zero-shot RNA tests -> results/rna.json
qprotbench.py   zero-shot protein tests -> results/protein.json
qxl.py          quill-xl2: experts, four sizes, evaluation -> results/xl.json
qxlexport.py    exports quill-xl2-nano for the browser
qxlparity.py    browser-parity reference for quill-xl2-nano
qverify4.py     checks every quill-xl2 number in the preprint
docs/           the web app illustrating the model, its engine, and its tests
models/         the weight files
results/        benchmark output and figures
```

## Running it

```sh
python3 qwritten.py                       # build the zero-data model (< 1 s)
python3 qmodel.py                         # assemble the counted models
python3 qbench.py                         # all benchmarks (~50 min on 2 cores)
python3 qbench.py lm                      # or one of: lm gen brca1 anno gb
python3 qmakefigs.py                      # figures
python3 qverify.py                        # check the preprint's numbers

python3 qgoparse.py                       # GO terms and proteins from the EMBL files
python3 qgo.py data && python3 qgo.py build
python3 qso.py data && python3 qso.py build
python3 qgobench.py && python3 qsobench.py
python3 qverify2.py                       # check the ontology numbers

pip install pyjaspar                       # JASPAR 2024, used by qnongenic.py
curl -L -o /home/claude/data/gox/go-basic.obo \
  https://github.com/geneontology/go-ontology/releases/latest/download/go-basic.obo
python3 qgox.py data && python3 qgox.py build
python3 qgoxbench.py && python3 qdomains.py
python3 qnongenic.py tf && python3 qnongenic.py reg && python3 qnongenic.py repeats
python3 qverify3.py                       # check the quill-xl numbers

pip install biopython ViennaRNA           # PROSITE parsing, RNA folding
git clone https://github.com/stevin-wilson/PrositePatternsToPythonRegex  # prosite.dat
python3 qprot.py cache && python3 qrna.py cache
python3 qprotbench.py && python3 qrna.py validate
python3 qxl.py build && python3 qxl.py bench && python3 qxl.py extra
python3 qxlexport.py nano && python3 qxlparity.py && node docs/xlparity.js
python3 qverify4.py                       # check the quill-xl2 numbers
```

`MODELS=quill-k python3 qbench.py lm` restricts a run to named models.

Browser build and its tests:

```sh
python3 qexport.py quill-k && python3 qexport.py quill-s
python3 qontexport.py && python3 qontexamples.py
python3 qparity.py && python3 qontparity.py   # references from held-out chromosomes
cd docs && npm install
node parity.js quill-k && node parity.js quill-s   # JS vs Python, exact
node ontparity.js                                  # the ontology models, to 1e-5
node smoke.js                                      # page builds, every model loads
```

The two engines agree to 1e-10 bits/base with identical Viterbi paths.
