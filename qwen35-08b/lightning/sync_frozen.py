# Verbatim frozen-implementation reuse. No semantics here; only extraction + pin check.
# Source: qwen35-08b/kaggle_qwen35_08b_ple_eval_a2_arb.ipynb cells
# c-hashing, c-reader, c-inject, c-ple. Any drift aborts before training.
import hashlib
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
NB = REPO / "kaggle_qwen35_08b_ple_eval_a2_arb.ipynb"
CELLS = ("c-hashing", "c-reader", "c-inject", "c-ple")
FORBID = ("class MemoryArbitration", "train_arb(", "build_calibration(")
def load_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = {c.get("id"): "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"}
    missing = [c for c in CELLS if c not in code]
    assert not missing, "frozen cells missing: %s" % missing
    return code
def blob_sha(code):
    h = hashlib.sha256()
    for c in CELLS:
        h.update(code[c].encode("utf-8"))
    return h.hexdigest()
def check_no_drift(code):
    whole = "\n".join(code[c] for c in CELLS)
    for s in ("splitmix64", "head_layout", "shift_right_ignore_eos", "ngram_indices",
              "SharedValueReader", "rms_norm", "ReaderInjection", "decoder_layers",
              "MountPLE", "CompactPLE", "weight_scale"):
        assert s in whole, "frozen symbol missing: %s" % s
    for s in FORBID:
        assert s not in whole, "training logic leaked into frozen cells: %s" % s
    return True
def emit_module(out, code):
    # Writes generated module so Studio runs identical bytes; header records notebook SHA.
    nb_sha = hashlib.sha256(NB.read_bytes()).hexdigest()
    parts = ["# GENERATED from %s sha=%s cells=%s. Do not hand-edit; rerun sync_frozen.py.\n"
             % (NB.name, nb_sha, ",".join(CELLS))]
    for c in CELLS:
        parts.append("# --- %s (verbatim) ---\n" + code[c] + "\n")
    body = "".join(parts).rstrip("\n") + "\n"  # canonical single trailing LF
    out.write_bytes(body.encode("utf-8"))  # bytes, not write_text: no CRLF translation
    return hashlib.sha256(body.encode("utf-8")).hexdigest(), nb_sha
def verify_transferred():
    # Studio has no notebook; verify the transferred module against its SHA manifest.
    here = Path(__file__).parent
    man = json.loads((here / "frozen_shas.json").read_text())
    body = (here / "frozen_generated.py").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    assert digest == man["module_sha256"], "frozen module drift in transfer"
    print("notebook sha=%s (pinned at transfer)" % man["notebook_sha256"])
    print("module sha=%s ok: transferred bytes match manifest, no drift" % digest)
    return digest
if __name__ == "__main__":
    if not NB.exists():
        verify_transferred()
    else:
        code = load_cells()
        check_no_drift(code)
        mod_sha, nb_sha = emit_module(Path(__file__).parent / "frozen_generated.py", code)
        print("notebook sha=%s" % nb_sha)
        print("frozen blob sha=%s" % blob_sha(code))
        print("module sha=%s" % mod_sha)
        (Path(__file__).parent / "frozen_shas.json").write_bytes(
            (json.dumps({"notebook_sha256": nb_sha, "frozen_blob_sha256": blob_sha(code),
                        "module_sha256": mod_sha, "cells": list(CELLS)}, indent=2).rstrip("\n") + "\n").encode("utf-8"))
        print("ok: frozen reuse pinned, no drift")
