# Launcher: inspect -> freeze -> qualify -> train -> eval. Training blocked until qualify passes.
import argparse
import json
import os
import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config
def stage_inspect(a):
    import studio_flow
    studio_flow.setup()
def stage_freeze(a):
    import studio_flow
    hf = os.environ.get("HF_TOKEN")
    assert hf, "set HF_TOKEN"
    tok = studio_flow.load_tokenizer(hf)
    studio_flow.freeze_stage(tok)
def stage_qualify(a):
    import studio_flow
    hf = os.environ.get("HF_TOKEN")
    assert hf, "set HF_TOKEN"
    tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
    import frozen_generated as _FZ
    root = studio_flow.proot()
    store = _FZ.CompactPLE(str(root / "compact"))
    studio_flow.qualify_stage(tok, model, inj, versions, ngram_fn, FZ, store, root / "canonical-v1")
def stage_train(a):
    import studio_flow
    hf = os.environ.get("HF_TOKEN")
    assert hf, "set HF_TOKEN"
    tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
    import frozen_generated as _FZ
    store = _FZ.CompactPLE(str(studio_flow.proot() / "compact"))
    print(studio_flow.train_legs(tok, model, inj, versions, ngram_fn, FZ, store), flush=True)
def stage_eval(a):
    import studio_flow
    hf = os.environ.get("HF_TOKEN")
    assert hf, "set HF_TOKEN"
    tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
    import frozen_generated as _FZ
    store = _FZ.CompactPLE(str(studio_flow.proot() / "compact"))
    studio_flow.load_canonical_arb(studio_flow.proot(), inj, __import__("arb_frozen"))
    print(json.dumps({str(k): v["val_nll"] for k, v in studio_flow.eval_milestones(tok, model, inj, versions, ngram_fn, FZ, store).items()}, indent=2), flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["inspect", "freeze", "qualify", "train", "eval", "all"])
    ap.add_argument("--persistent", default=None)
    a = ap.parse_args()
    if a.persistent:
        os.environ[config.PERSISTENT_ENV] = a.persistent
    if a.stage == "all":
        stage_inspect(a)
    else:
        {"inspect": stage_inspect, "freeze": stage_freeze, "qualify": stage_qualify,
         "train": stage_train, "eval": stage_eval}[a.stage](a)
