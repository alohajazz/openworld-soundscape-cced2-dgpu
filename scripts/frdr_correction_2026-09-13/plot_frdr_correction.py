"""Regenerate Figure 3 from hash-verified public aggregates, without model fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path,
        default=HERE.parents[1]/'paper_artifacts/frdr_correction_2026-09-13')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.artifact_root.resolve(), args.output_dir.resolve()
    if output.exists() or root == output or root in output.parents:
        parser.error('--output-dir must be a new directory outside the frozen artifact tree')
    manifest = json.loads((HERE/'artifact_manifest.json').read_text())
    for rel, expected in manifest['approved_artifact_sha256'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == expected, rel
    t2 = read_csv(root/'results/table2/frdr_quiet_promoter_union_supervised.csv')
    fusion = read_csv(root/'results/fusion/fusion.csv')
    assert len(t2) == 3 and len(fusion) == 11
    assert [r['mode'].split()[0] for r in t2] == ['Quiet','Promoter','Union']
    best = max(fusion,key=lambda r:float(r['Recall']))  # first maximum, same retained alpha selection
    points = {name:{k:float(row[k]) for k in ['Recall','FP_h','TP','FP']}
              for name,row in zip(['Quiet','Promoter','Union','Fusion'], t2+[best])}
    for row in points.values():
        assert math.isclose(row['Recall'], row['TP']/1157, abs_tol=1e-14)
        assert math.isclose(row['FP_h'], row['FP']/(451/18), abs_tol=1e-13)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.2,5.4))
    formats={'Quiet':('o','#4C78A8',(-10,21),'right'),
             'Promoter':('D','#F58518',(12,-27),'left'),
             'Union':('^','#54A24B',(-12,14),'right'),
             'Fusion':('s','#B279A2',(12,21),'left')}
    for name,v in points.items():
        marker,colour,offset,alignment=formats[name]
        ax.scatter(v['FP_h'],v['Recall'],s=120,marker=marker,color=colour,edgecolor='black',linewidth=.8,zorder=3,label=name)
        ax.annotate(f"{name}\nrecall {v['Recall']:.3f}",(v['FP_h'],v['Recall']),xytext=offset,textcoords='offset points',ha=alignment,fontsize=9)
    ax.axvline(10,color='0.5',linestyle='--',linewidth=1,label='Target FP/h = 10')
    x=[v['FP_h'] for v in points.values()]
    ax.set_xlim(min(x)-.15,max(x)+.25); ax.set_ylim(0,.55)
    ax.set_xlabel('False positives per hour (FP/h)'); ax.set_ylabel('Recall')
    ax.grid(alpha=.25)
    ax.set_title('FRDR right-whale upcalls: FP/h-constrained operating points')
    ax.legend(loc='center right',frameon=True); fig.tight_layout()
    output.mkdir(parents=True)
    fig.savefig(output/'Fig3_FRDR_all50.png',dpi=300,bbox_inches='tight')
    fig.savefig(output/'Fig3_FRDR_all50.pdf',bbox_inches='tight')
    plt.close(fig)
    (output/'operating_points.json').write_text(json.dumps(points,indent=2)+'\n')
    print('PASS: Figure 3 regenerated from verified aggregates; no model or threshold sweep rerun')

if __name__ == '__main__': main()
