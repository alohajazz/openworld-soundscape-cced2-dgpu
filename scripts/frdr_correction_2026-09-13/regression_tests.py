"""Executable regression checks; load source AST helpers without module side effects."""
import ast
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))
from frdr_file_universe import complete_annotations

def helpers(path):
    tree = ast.parse(path.read_text())
    ns = dict(np=np, pd=pd, K_CONSEC=2, GAP_SEC=3., TOL_SEC=10., SMOOTH_WIN=3, TARGET_FP=10.)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    return ns, {n.name: ast.dump(n, include_attributes=False) for n in functions}

def run():
    checks = []
    m = pd.DataFrame({"base": ["positive.wav", "negative.wav"]})
    raw = {"positive.wav": [5., 80.]}
    complete = complete_annotations(m, raw)
    assert complete == {"positive.wav": [5., 80.], "negative.wav": []}
    try:
        complete_annotations(m, {"outside.wav": [1.]})
    except ValueError:
        checks.append("annotation outside manifest fails closed")
    else:
        raise AssertionError("outside-manifest annotation accepted")
    cache = {b: (np.array([5., 7., 20., 22.]), np.array([2., 2., 0., 0.])) for b in m.base}
    for source in sorted((ROOT / "source").glob("*.py")):
        if source.name == "frdr_file_universe.py":
            continue
        text = source.read_text().replace("from frdr_file_universe import complete_annotations\n", "")
        text = "\n".join(line for line in text.split("\n") if " = complete_annotations(m, " not in line)
        if source.name == "frdr_label_efficiency_param.py":
            text = text.replace('OUTDIR = Path(_os.environ.get("LE_OUT", "/workspace/logs/labeleff_"+_os.environ.get("LE_TAG","x")))', 'OUTDIR = Path("/workspace/logs/labeleff_"+_os.environ.get("LE_TAG","x"))')
        assert text == (ROOT / "original" / source.name).read_text(), source.name + ": unrelated source change"
        ns, nodes = helpers(source)
        old, oldnodes = helpers(ROOT / "original" / source.name)
        assert nodes == oldnodes, source.name + ": original helper/model functions changed"
        before = old["eval_at"](cache, raw, 1., 1.)
        after = ns["eval_at"](cache, complete, 1., 1.)
        assert before[2] == after[2] == 1 and after[3] == before[3] + 1
        assert before[0] == after[0] == .5
        if len(after) == 6:
            assert before[4] == after[4] == 1
        assert ns["eval_at"](cache, complete, 1., 1.) == old["eval_at"](cache, complete, 1., 1.)
        # Exact threshold, run gap <=3, minimum two points, and ±10-s matching.
        ev = ns["extract_events"](np.array([0., 3., 7., 9.]), np.ones(4), 1.)
        assert len(ev) == 2
        matcher = ns.get("match_1to1", ns.get("match"))
        assert matcher([0., 30.], [10., 20.], 10.) == (2, 0, 0)
        assert matcher([0.], [10.00001], 10.) == (0, 1, 1)
        if "eval_union" in ns:
            q = {"positive.wav": (np.array([0., 2.]), np.array([2., 1.]))}
            for separation, expected in [(2.9999, 1), (3., 2)]:
                p = {"positive.wav": (np.array([separation, separation+2]), np.array([9., 8.]))}
                r = ns["eval_union"](q, p, {"positive.wav": []}, 1., 1., 1.)
                assert r[3] == expected and r == old["eval_union"](q, p, {"positive.wav": []}, 1., 1., 1.)
                if separation < 3:
                    # Earlier quiet event wins despite promoter's higher peak.
                    assert ns["eval_union"](q, p, {"positive.wav": [-9.]}, 1., 1., 1.)[2] == 1
        checks.append(source.name + ": empty-file FP, TP/FN, all-annotated equivalence, event boundaries, helpers unchanged")
    print("PASS", len(checks), "regression groups")
    return checks

if __name__ == "__main__":
    run()
