"""Figures for the preprint. Reads results/*.json and the saved score arrays."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
RES = Path('/home/claude/quill/results')
FIG = RES / 'figs'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.7,
                     'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 200, 'savefig.dpi': 200, 'legend.frameon': False})
INK, ACC, ACC2, WARN, GREY = '#1c2430', '#0d9488', '#2563eb', '#d97706', '#94a3b8'


def fig_scaling():
    d = json.load(open(RES / 'scaling.json'))
    orders = d['orders']
    params = [4 ** k * 4 for k in orders]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for key, lab, col, mk in (('human_chr20', 'human chr20 (held out)', ACC2, 'o'),
                              ('human_chr21', 'human chr21 (held out)', '#60a5fa', 's'),
                              ('mouse_chr19', 'mouse chr19 (held-out species)', WARN, '^')):
        ax[0].plot(params, d[key], marker=mk, ms=3.4, lw=1.3, color=col, label=lab)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('context-model parameters')
    ax[0].set_ylabel('bits per base')
    ax[0].set_title('a   context model, and the three models', loc='left', fontsize=9,
                    fontweight='bold')
    ax[0].grid(alpha=.18, lw=.5)
    # the three assembled models, at their own parameter counts
    bench = json.load(open(RES / 'bench.json'))['lm']
    pts = [('quill-k', 10252, '#0f766e', 'o'), ('quill-s', 1406988, '#0d9488', 's'),
           ('quill-l', 67467276, '#5eead4', '^')]
    for nm, npar, col, mk in pts:
        hv = bench['human_heldout_chr19_22'][nm]['bits_per_base']
        mv = bench['mouse_chr19_heldout_species'][nm]['bits_per_base']
        ax[0].plot([npar], [hv], marker=mk, ms=6, color=col, mec=INK, mew=.6,
                   ls='none', zorder=5)
        ax[0].plot([npar], [mv], marker=mk, ms=6, color=col, mec=WARN, mew=.9,
                   ls='none', mfc='none', zorder=5)
        ax[0].annotate(nm, (npar, hv), textcoords='offset points', xytext=(7, -3),
                       fontsize=6.4, ha='left', color=INK)
    ax[0].plot([], [], marker='o', ls='none', color=INK, ms=4,
               label='full grammar, human (filled)')
    ax[0].plot([], [], marker='o', ls='none', mfc='none', mec=WARN, ms=4,
               label='full grammar, mouse (open)')
    ax[0].set_xlim(40, 4.0e8)
    ax[0].legend(fontsize=6.2, loc='lower left')
    gains = [d[k][0] - d[k][-1] for k in ('human_chr20', 'human_chr21', 'mouse_chr19')]
    bars = ax[1].barh(['human\nchr20', 'human\nchr21', 'mouse\nchr19'], gains,
                      color=[ACC2, '#60a5fa', WARN], height=.6)
    for b, g in zip(bars, gains):
        ax[1].text(g + .006, b.get_y() + b.get_height() / 2, f'{g:.3f}', va='center', fontsize=8)
    ax[1].set_xlabel('bits per base gained, order 2 → order 12')
    ax[1].set_xlim(0, max(gains) * 1.35)
    ax[1].set_title('b   what the extra 67M parameters buy', loc='left', fontsize=9, fontweight='bold')
    ax[1].grid(axis='x', alpha=.18, lw=.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_scaling.png', bbox_inches='tight')
    print('fig1', gains)


def fig_brca1():
    b = json.load(open(RES / 'bench.json'))['brca1']
    csq_order = ['Nonsense', 'Canonical splice', 'Splice region', 'Missense', 'Synonymous',
                 'Intronic']
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=.75, wspace=.55)

    ax = fig.add_subplot(gs[0, 0])
    rates = [b['lof_rate_by_consequence'][c] for c in csq_order]
    ns = [b['consequence_counts'][c] for c in csq_order]
    ax.barh(range(len(csq_order)), rates, color=[WARN if r > .5 else GREY for r in rates], height=.6)
    ax.set_yticks(range(len(csq_order)))
    ax.set_yticklabels([f'{c}  ({n:,})' for c, n in zip(csq_order, ns)], fontsize=7)
    ax.invert_yaxis()
    for i, r in enumerate(rates):
        ax.text(r + .03, i, f'{r:.0%}', va='center', fontsize=7)
    ax.set_xlim(0, 1.25); ax.set_xlabel('fraction loss-of-function', fontsize=8)
    ax.set_title('a   what the assay measures', loc='left', fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=.18, lw=.5)

    ax = fig.add_subplot(gs[0, 1])
    sm = b['quill-k_splice_matrix']
    keys = [c for c in csq_order if f'auroc_{c}' in sm]
    vals = [sm[f'auroc_{c}'] for c in keys]
    ax.barh(range(len(keys)), vals, color=[ACC if v > .6 else GREY for v in vals], height=.6)
    ax.axvline(.5, color=INK, lw=.8, ls='--')
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + .02, i, f'{v:.2f}', va='center', fontsize=7)
    ax.set_xlim(0, 1.18); ax.set_xlabel('AUROC, written splice matrices alone', fontsize=8)
    ax.set_title('b   where 176 written parameters are enough', loc='left', fontsize=9,
                 fontweight='bold')
    ax.grid(axis='x', alpha=.18, lw=.5)

    ax = fig.add_subplot(gs[1, :])
    groups = [('all SNVs\n(3,893)', 'all'),
              ('noncoding\n(1,125)', 'noncoding'),
              ('splice-region\n+ intronic (1,078)', 'splice_and_intronic'),
              ('missense\n(2,086)', 'auroc_Missense')]
    series = [('quill-k, written splice matrices (176 params)', b['quill-k_splice_matrix'], '#99f6e4'),
              ('quill-s, counted splice matrices (176 params)', b['quill-s_splice_matrix'], ACC),
              ('quill-s, whole-window Δ (Evo 2 protocol)', b['quill-s_window_delta'], '#7dd3c0'),
              ('phyloP, mammalian', b['phyloP_mammalian'], GREY),
              ('CADD v1.3 (supplied with the assay)', b['CADD'], ACC2)]
    w = .16
    xs = np.arange(len(groups))
    for j, (lab, src, col) in enumerate(series):
        vals = [src.get(k, np.nan) for _, k in groups]
        ax.bar(xs + (j - 2.0) * w, vals, w * .9, label=lab, color=col, zorder=3)
        for x, v in zip(xs + (j - 2.0) * w, vals):
            if not np.isnan(v):
                ax.text(x, v + .015, f'{v:.2f}', ha='center', fontsize=6.5, zorder=4)
    # published zero-shot values; not re-run here (see Methods)
    pub = [(0, .73, 'Evo 2 7B 0.73', '#7f1d1d', (0, (3, 2)), 'left', .018),
           (1, .87, 'Evo 2 7B 0.87', '#7f1d1d', (0, (3, 2)), 'left', .018),
           (1, .77, 'GPN-MSA 0.77', '#a16207', (0, (1, 1.6)), 'left', -.042),
           (3, .57, 'Evo 2 7B 0.57', '#7f1d1d', (0, (3, 2)), 'right', .018)]
    for gi, v, lab, col, ls, ha, dy in pub:
        ax.plot([gi - .46, gi + .46], [v, v], color=col, lw=1.3, ls=ls, zorder=5)
        xt = gi - .45 if ha == 'left' else gi + .45
        ax.text(xt, v + dy, lab, fontsize=6.3, color=col, ha=ha, zorder=6,
                bbox=dict(fc='white', ec='none', pad=.6, alpha=.82))
    ax.axhline(.5, color=INK, lw=.8, ls='--', zorder=1)
    ax.text(3.48, .52, 'chance', fontsize=6.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([g for g, _ in groups], fontsize=7.5)
    ax.set_ylim(0, 1.16); ax.set_ylabel('AUROC')
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.legend(fontsize=6.6, ncol=2, loc='lower center', bbox_to_anchor=(.5, -.46))
    ax.grid(axis='y', alpha=.18, lw=.5, zorder=0)
    ax.set_title('c   published values are dashed; the gap to the large models is missense',
                 loc='left', fontsize=9, fontweight='bold')
    fig.savefig(FIG / 'fig2_brca1.png', bbox_inches='tight')
    print('fig2 ok')


GENETIC = {}
_t = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
_k = 0
for _a in 'TCAG':
    for _b in 'TCAG':
        for _c in 'TCAG':
            GENETIC[_a + _b + _c] = _t[_k]; _k += 1


def fig_generation():
    import qstates as S
    import qinfer as I
    from qbench import Model
    g = json.load(open(RES / 'bench.json'))['gen']
    z = np.load('/home/claude/data/tables/anno_train.npz')
    stops = [3 * 16 + 0 * 4 + 0, 3 * 16 + 2 * 4 + 0, 3 * 16 + 0 * 4 + 2]
    names = [a + b + c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
    cu_ref = z['codon'].astype(float); cu_ref[stops] = 0; cu_ref /= cu_ref.sum()
    m = Model('quill-l')
    # 2 Mb of prompted generation, so the codon counts are not noise-limited
    cods = []
    compact = None
    for r in range(100):
        seq, path = I.sample(20000, 4000 + r, 1, *m.args(), S.G.idx['UTR5'], 1.0, 1024)
        c0 = np.flatnonzero(path == S.G.idx['CDS0']); c0 = c0[c0 + 2 < len(seq)]
        if len(c0):
            cod = seq[c0[:, None] + np.arange(3)]
            cods.append(cod[:, 0] * 16 + cod[:, 1] * 4 + cod[:, 2])
        if compact is None:
            atg = np.flatnonzero(path == S.G.idx['ATG0'])
            stp = np.flatnonzero(path == S.G.idx['STOP0'])
            if len(atg) and len(stp) and stp[-1] > atg[0] and stp[-1] - atg[0] < 9000:
                compact = (seq, path, max(0, int(atg[0]) - 700),
                           min(len(path), int(stp[-1]) + 1800))
        if sum(len(c) for c in cods) > 120000 and compact is not None:
            break
    ci = np.concatenate(cods)
    cu = np.bincount(ci, minlength=64).astype(float)
    cu[stops] = 0; cu /= cu.sum()
    rcod = np.corrcoef(cu_ref, cu)[0, 1]

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, .78], hspace=.78, wspace=.33)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(cu_ref * 1000, cu * 1000, s=12, color=ACC, zorder=3, edgecolor='none')
    lim = max(cu_ref.max(), cu.max()) * 1000 * 1.1
    ax.plot([0, lim], [0, lim], color=GREY, lw=.8, ls='--')
    for i2 in np.argsort(-cu_ref)[:4]:
        ax.annotate(names[i2], (cu_ref[i2] * 1000, cu[i2] * 1000), fontsize=6.5,
                    xytext=(3, -1), textcoords='offset points')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('measured human usage (per 1,000)', fontsize=8)
    ax.set_ylabel('Quill-L generated', fontsize=8)
    ax.set_title(f'a   codon usage, r = {rcod:.3f}  ({len(ci):,} codons)',
                 loc='left', fontsize=9, fontweight='bold')
    ax.grid(alpha=.18, lw=.5)

    ax = fig.add_subplot(gs[0, 1])
    items = [('shuffled chr20', g['shuffled_control']['kmer6_js'], GREY),
             ('Quill-S', g['quill-s']['kmer6_js'], ACC),
             ('Quill-L', g['quill-l']['kmer6_js'], '#0f766e'),
             ('real vs real\n(chr20 halves)', g['reference_second_half']['kmer6_js'], ACC2)]
    ax.barh([i2[0] for i2 in items], [i2[1] for i2 in items],
            color=[i2[2] for i2 in items], height=.6)
    for i2, (lab, v, c) in enumerate(items):
        ax.text(v + .003, i2, f'{v:.4f}', va='center', fontsize=7.5)
    ax.set_xlabel('6-mer Jensen–Shannon divergence\nfrom real chr20', fontsize=8)
    ax.set_xlim(0, max(i2[1] for i2 in items) * 1.35)
    ax.set_title('b   composition realism', loc='left', fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=.18, lw=.5)

    ax = fig.add_subplot(gs[1, :])
    RIB = {'intergenic': '#94a3b8', 'cpg_island': '#14b8a6', 'tandem': '#f59e0b',
           'utr5': '#60a5fa', 'cds': '#10b981', 'intron': '#dbe3ec', 'splice': '#a78bfa',
           'utr3': '#38bdf8', 'ncrna': '#f472b6', 'polya': '#fb923c', 'start': '#15803d',
           'stop': '#dc2626'}

    def rib(nm):
        if nm.startswith('CDS'): return 'cds'
        if nm.startswith('INTRON'): return 'intron'
        if nm.startswith(('DON', 'ACC')): return 'splice'
        if nm.startswith('PAS'): return 'polya'
        if nm.startswith('ATG'): return 'start'
        if nm.startswith('STOP'): return 'stop'
        return {'IG': 'intergenic', 'CGI': 'cpg_island', 'TANDEM': 'tandem', 'UTR5': 'utr5',
                'UTR3': 'utr3', 'NCRNA': 'ncrna'}.get(nm, 'intergenic')
    seq, path, a, b = compact
    sub = path[a:b]
    prev, start = None, 0
    for i2 in range(len(sub) + 1):
        cur = rib(S.G.name[sub[i2]]) if i2 < len(sub) else None
        if cur != prev:
            if prev is not None:
                ax.add_patch(Rectangle((start, 0), i2 - start, 1, color=RIB[prev], lw=0))
            prev, start = cur, i2
    ax.set_xlim(0, len(sub)); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('position in a sampled sequence (bp)', fontsize=8)
    seen = []
    for i2 in range(len(sub)):
        r2 = rib(S.G.name[sub[i2]])
        if r2 not in seen:
            seen.append(r2)
    hs = [Rectangle((0, 0), 1, 1, color=RIB[s2]) for s2 in seen]
    ax.legend(hs, [s2.replace('_', ' ') for s2 in seen], fontsize=7, ncol=7,
              loc='upper center', bbox_to_anchor=(.5, -.32), handlelength=1.2)
    ax.set_title('c   one gene the model wrote, coloured by the state it was writing from',
                 loc='left', fontsize=9, fontweight='bold')
    fig.savefig(FIG / 'fig3_generation.png', bbox_inches='tight')
    print('fig3 ok, codon r =', round(float(rcod), 4), 'codons', len(ci),
          'gene span', b - a)


def fig_ontology():
    """The two ontology layers: where written weights win, and where they fail."""
    G = json.load(open(RES / 'gobench.json'))
    S = json.load(open(RES / 'sobench.json'))
    fig, ax = plt.subplots(1, 4, figsize=(13.4, 3.3))

    # (a) family AUROC, written vs counted, against how many examples the counted arm had
    fam_k = G['classify']['fam/quill-k']['per_class_auroc']
    fam_s = G['classify']['fam/quill-s']['per_class_auroc']
    ntrain = {'olfactory_receptor': 348, 'zinc_finger': 217, 'immunoglobulin_V': 104,
              'histone': 84, 'ribosomal_protein': 56, 'collagen': 37, 'keratin': 24}
    order = sorted(fam_k, key=lambda k: -ntrain[k])
    xs = np.arange(len(order))
    ax[0].plot(xs, [fam_k[k] for k in order], 'o-', color=ACC, lw=1.4, ms=5,
               label='written (no examples)', zorder=4)
    ax[0].plot(xs, [fam_s[k] for k in order], 's--', color=ACC2, lw=1.2, ms=4.6,
               label='counted', zorder=3)
    for i, k in enumerate(order):
        ax[0].annotate(f'n={ntrain[k]}', (i, min(fam_k[k], fam_s[k])),
                       textcoords='offset points', xytext=(0, -11), fontsize=5.8,
                       ha='center', color=GREY)
    short = {'olfactory_receptor': 'olfactory', 'zinc_finger': 'zinc finger',
             'immunoglobulin_V': 'Ig V', 'histone': 'histone',
             'ribosomal_protein': 'ribosomal', 'collagen': 'collagen',
             'keratin': 'keratin'}
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([short[k] for k in order], fontsize=6.4, rotation=32,
                          ha='right')
    ax[0].set_ylim(0.80, 1.02); ax[0].set_ylabel('AUROC', fontsize=8)
    ax[0].set_title('a   gene family from coding sequence', loc='left', fontsize=9,
                    fontweight='bold')
    ax[0].legend(fontsize=6.6, loc='lower left'); ax[0].grid(alpha=.18, lw=.5, axis='y')

    # (b) the written boxes are present where they should be
    bx = S['boxes']
    keys = list(bx)
    xs = np.arange(len(keys)); w = 0.26
    for j, (lab, fld, col) in enumerate((('in true members', 'mean_in_members', ACC),
                                         ('same bases, shuffled', 'mean_in_shuffled', GREY),
                                         ('other RNA classes', 'mean_in_other_classes', WARN))):
        ax[1].bar(xs + (j - 1) * w, [bx[k][fld] for k in keys], w * .92, color=col,
                  label=lab, zorder=3)
    ax[1].axhline(0, color=INK, lw=.7)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([k.split('/')[1].replace('BOX_', '').replace('_', ' ')
                           for k in keys], fontsize=6.4)
    ax[1].set_ylabel('log2 odds of the written box', fontsize=8)
    ax[1].set_title('b   the written boxes are really there', loc='left', fontsize=9,
                    fontweight='bold')
    ax[1].legend(fontsize=6.4, loc='upper right'); ax[1].grid(alpha=.18, lw=.5, axis='y')

    # (c) generated sequence carries the written architecture
    gen = G['generate']
    col = gen['fam/collagen']['glycine_by_triplet_phase']
    znf = gen['fam/zinc_finger']
    ker = gen['fam/keratin']
    groups = [('collagen\nglycine', col[0], float(np.mean(col[1:])), 'phase 0', 'other'),
              ('zinc finger\ncysteine', znf['cysteine_at_anchored_positions'],
               znf['cysteine_elsewhere'], 'anchored', 'other'),
              ('zinc finger\nhistidine', znf['histidine_at_anchored_positions'],
               znf['histidine_elsewhere'], 'anchored', 'other'),
              ('keratin\nhydrophobic', ker['hydrophobic_at_a_and_d'],
               ker['hydrophobic_elsewhere'], 'a and d', 'other')]
    xs = np.arange(len(groups)); w = 0.36
    ax[2].bar(xs - w / 2, [g[1] for g in groups], w, color=ACC, zorder=3,
              label='at the written position')
    ax[2].bar(xs + w / 2, [g[2] for g in groups], w, color=GREY, zorder=3,
              label='elsewhere in the repeat')
    for i, g in enumerate(groups):
        ax[2].annotate(f'{g[1]:.2f}', (i - w / 2, g[1]), textcoords='offset points',
                       xytext=(0, 2), fontsize=6, ha='center', color=INK)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([g[0] for g in groups], fontsize=6.4)
    ax[2].set_ylabel('fraction of residues', fontsize=8); ax[2].set_ylim(0, 1.0)
    ax[2].set_title('c   sampled sequence keeps the period', loc='left', fontsize=9,
                    fontweight='bold')
    ax[2].legend(fontsize=6.4, loc='upper right'); ax[2].grid(alpha=.18, lw=.5, axis='y')

    # (d) the boundary: motif-as-mechanism versus motif-as-one-signal
    mi, ce = S['minor_intron'], S['cis_elements']
    items = [('snoRNA box C', bx['snoRNA_CD/BOX_C']['auroc_vs_shuffled'], ACC),
             ('snoRNA box ACA', bx['snoRNA_HACA/BOX_ACA']['auroc_vs_shuffled'], ACC),
             ('snRNA Sm site', bx['snRNA/SM_SITE']['auroc_vs_shuffled'], ACC),
             ('U12 branch point', mi['auroc_branch_only'], ACC),
             ('GC box (SP1)', ce['GC_BOX']['auroc_vs_shuffled_same_window'], WARN),
             ('initiator', ce['INR']['auroc_vs_shuffled_same_window'], WARN),
             ('TCT element', ce['TCT']['auroc_vs_shuffled_same_window'], WARN),
             ('CCAAT box', ce['CCAAT']['auroc_vs_shuffled_same_window'], WARN),
             ('TATA box', ce['TATA']['auroc_vs_shuffled_same_window'], WARN)]
    ys = np.arange(len(items))[::-1]
    ax[3].barh(ys, [i[1] for i in items], 0.62, color=[i[2] for i in items], zorder=3)
    ax[3].axvline(0.5, color=INK, lw=.8, ls=(0, (3, 2)))
    for y, it in zip(ys, items):
        ax[3].annotate(f'{it[1]:.3f}', (it[1], y), textcoords='offset points',
                       xytext=(4, -2.4), fontsize=6.2, color=INK)
    ax[3].set_yticks(ys); ax[3].set_yticklabels([i[0] for i in items], fontsize=6.6)
    ax[3].set_xlim(0, 1.06)
    ax[3].set_xlabel('AUROC against a shuffle of the same bases', fontsize=7.6)
    ax[3].set_title('d   bound, or one signal of many', loc='left', fontsize=9,
                    fontweight='bold')
    ax[3].grid(alpha=.18, lw=.5, axis='x')

    fig.tight_layout()
    p = RES / 'figs' / 'fig4_ontology.png'
    fig.savefig(p, dpi=190); plt.close(fig)
    print(f'wrote {p}')
