# Studio end-to-end flow. Order: setup -> freeze -> qualify -> train -> eval.
# Training never runs before the qualification gate passes. All checkpoints persist.
import json
import os
import sys
import time
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config
T0 = time.perf_counter()
HARD_S = 3 * 3600 + 30 * 60 + 600
def need_token():
    t = os.environ.get("HF_TOKEN")
    assert t, "set HF_TOKEN in the Studio environment"
    return t
def proot():
    r = Path(os.environ.get(config.PERSISTENT_ENV, config.DEFAULT_PERSISTENT))
    r.mkdir(parents=True, exist_ok=True)
    return r
def setup():
    import sync_frozen
    if not sync_frozen.NB.exists():
        sync_frozen.verify_transferred()
    else:
        code = sync_frozen.load_cells()
        sync_frozen.check_no_drift(code)
        mod_sha, nb_sha = sync_frozen.emit_module(HERE / "frozen_generated.py", code)
        print("frozen ok nb=%s mod=%s" % (nb_sha[:16], mod_sha[:16]), flush=True)
    import ple
    r = proot()
    info = ple.disks(r, os.environ.get(config.SCRATCH_ENV, "/tmp"))
    (r / "disk-report.json").write_text(json.dumps(info, indent=2))
    mode = ple.plan(info)
    (r / "storage-plan.json").write_text(json.dumps({"mode": mode, "disks": info}, indent=2))
    print("setup ok mode=%s" % mode, flush=True)
    return mode
def load_tokenizer(hf_token):
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    trev = HfApi(token=hf_token).model_info(config.REVS["target_id"]).sha
    assert trev == config.REVS["tokenizer_rev"], "tokenizer rev drift"
    return AutoTokenizer.from_pretrained(config.REVS["target_id"], token=hf_token, revision=trev, trust_remote_code=True)
def load_stack(hf_token):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
    trev = HfApi(token=hf_token).model_info(config.REVS["target_id"]).sha
    assert trev == config.REVS["tokenizer_rev"], "tokenizer rev drift"
    srev = HfApi(token=hf_token).model_info(config.REVS["source_id"]).sha
    assert srev.startswith(config.REVS["source_rev_prefix"]), "source rev drift"
    tok = AutoTokenizer.from_pretrained(config.REVS["target_id"], token=hf_token, revision=trev, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(config.REVS["target_id"], token=hf_token, revision=trev, trust_remote_code=True)
    assert cfg.text_config.hidden_size == 1024 and cfg.text_config.num_hidden_layers == 24
    sys.path.insert(0, str(HERE))
    import frozen_generated as FZ
    import types as _types
    # The frozen cells reference the Kaggle config object C (c-config cell, not
    # extracted). Provide the same pins from config.py without touching frozen bytes.
    FZ.C = _types.SimpleNamespace(N_LAYERS=24, EOS=config.ADDR["eos"], VOCAB=config.ADDR["vocab"],
                                  NGRAM=config.ADDR["ngram"], HEADS_PER_NGRAM=config.ADDR["heads_per_ngram"],
                                  VOCAB_BASE=config.ADDR["vocab_base"], SEED=config.ADDR["seed"],
                                  ROW_DIM=160, ROWS_PER_PART=config.ADDR["rows_per_part"])
    model = None
    for dt in (torch.float16, torch.float32):
        try:
            m = AutoModelForCausalLM.from_pretrained(config.REVS["target_id"], revision=trev, token=hf_token,
                                                     trust_remote_code=True, device_map={"": 0}, torch_dtype=dt, low_cpu_mem_usage=True)
            m.eval()
            m.requires_grad_(False)
            ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids[:, :64].cuda()
            with torch.inference_mode():
                lg = m(input_ids=ids, use_cache=False).logits
            assert torch.isfinite(lg.float()).all()
            model = m
            print("backbone ok dtype=%s" % str(dt), flush=True)
            del ids, lg
            break
        except Exception as e:
            print("%s rejected: %s" % (str(dt), str(e)[:200]), flush=True)
    assert model is not None
    inj = FZ.ReaderInjection(model, [2, 8], 2560, 1024, 1, 1e-3).to("cuda")
    versions = {n: p._version for n, p in model.named_parameters()}
    def ngram_fn(ids):
        return FZ.ngram_indices(ids, eos_token_id=config.ADDR["eos"], vocab_size=config.ADDR["vocab"],
                                ngram_size=config.ADDR["ngram"], heads_per_ngram=config.ADDR["heads_per_ngram"],
                                vocab_size_base=config.ADDR["vocab_base"], ple_layer_index=0, seed=config.ADDR["seed"])
    return tok, model, inj, versions, ngram_fn, FZ
def load_canonical_arb(root, inj, arb_mod):
    import torch
    refdir = Path(root) / "canonical-v1"
    assert (refdir / "arbitration-c.pt").exists(), "copy Kaggle arbitration-c.pt/json + alpha8-stats.json + summary.json to %s first" % refdir
    blob = torch.load(str(refdir / "arbitration-c.pt"), map_location="cpu")
    arb = arb_mod.MemoryArbitration(hidden=1024).to("cuda")
    arb.w.data.copy_(blob["w"].to("cuda"))
    arb.b.data.copy_(blob["b"].to("cuda"))
    arb.alpha2_raw.data.copy_(blob["alpha2_raw"].to("cuda"))
    assert abs(float(arb.alpha2().detach()) - config.CANON["alpha2_frozen"]) < 1e-6
    arb.requires_grad_(False)
    from safetensors.torch import load_file
    ck = Path(root) / "ckpts" / config.CANON["base_reader_file"]
    import hashlib as _hl
    h = _hl.sha256(ck.read_bytes()).hexdigest()
    assert h == config.CANON["base_reader_sha256"], "base reader bytes drift"
    st = load_file(str(ck), device="cpu")
    inj.load_state_dict({k: v.to("cuda") for k, v in st.items()})
    inj.requires_grad_(False)
    g2 = float(inj.readers["2"].gamma.detach().cpu())
    g8 = float(inj.readers["8"].gamma.detach().cpu())
    assert abs(g2 - config.CANON["gamma2"]) < 1e-6 and abs(g8 - config.CANON["gamma8"]) < 1e-6
    return arb, refdir
def freeze_stage(tok):
    import calibration
    import frozen_data
    import ple
    import torch
    r = proot()
    hf_token = need_token()
    man = calibration.build_all(tok, hf_token, r / "calib")
    full, _ = __import__("train").ordered_full(r / "calib")
    val_full = frozen_data.load_blocks(r / "frozen-v1" / "tokens-full.uint32le", config.REVS["val_full_sha256"]) if (r / "frozen-v1" / "tokens-full.uint32le").exists() else None
    print("freeze ok calib=%s val_cached=%s" % (man["milestones"]["1000"]["sha256"][:16], val_full is not None), flush=True)
    return man
def _qual_inputs():
    import frozen_data
    r = proot()
    hs_rows, _ = frozen_data.hellaswag_rows(1000, need_token())
    lam_rows, _ = frozen_data.lambada_rows(1000, need_token())
    val_full = frozen_data.load_blocks(r / "frozen-v1" / "tokens-full.uint32le", config.REVS["val_full_sha256"])
    doms = {d: frozen_data.load_blocks(r / "frozen-v1" / ("tokens-%s.uint32le" % d), config.REVS["domains"][d]) for d in frozen_data.DOMAINS}
    return hs_rows, lam_rows, val_full, doms
def qualify_gate1(tok, model, inj, versions, ngram_fn, FZ, store):
    # Gate 1 (absolute): static arm A reproduces persisted v40 within cross-backend tol.
    import json as _json
    import torch as _torch
    import arb_frozen
    import evaluate
    r = proot()
    hs_rows, lam_rows, val_full, doms = _qual_inputs()
    v40 = _json.loads((r / "v40" / "summary.json").read_text())
    v40cal = v40["calibrated"]
    assert v40cal["mult"] == [1.25, 0.125]
    v40hs = [_json.loads(x) for x in (r / "v40" / "hellaswag-calibrated.jsonl").read_text().splitlines() if x.strip()]
    v40lam = [_json.loads(x) for x in (r / "v40" / "lambada-calibrated.jsonl").read_text().splitlines() if x.strip()]
    arb_probe, _ = load_canonical_arb(r, inj, arb_frozen)  # pins base reader SHA + gammas
    del arb_probe
    g2 = float(inj.readers["2"].gamma.detach().cpu())
    g8 = float(inj.readers["8"].gamma.detach().cpu())
    t = config.QUAL_TOL
    with _torch.no_grad():
        inj.readers["2"].gamma.copy_(_torch.tensor(float(g2 * 1.25)))
        inj.readers["8"].gamma.copy_(_torch.tensor(float(g8 * 0.125)))
    try:
        fa = evaluate.eval_all_static(model, inj, store, ngram_fn, tok, val_full, doms, hs_rows, lam_rows, T0, HARD_S)
    finally:
        with _torch.no_grad():
            inj.readers["2"].gamma.copy_(_torch.tensor(float(g2)))
            inj.readers["8"].gamma.copy_(_torch.tensor(float(g8)))
    # Cross-backend rule (predeclared): manifests align exactly (checked in diag);
    # fp16 kernels may flip razor-close argmax calls, so per-example exactness is
    # replaced by: flips <= 2/1000 AND every flip margin-close (< 0.5 nats; a dead
    # or miswired hook shifts logits O(1+) and NLL-means O(1e-2+), far above this).
    # Aggregates keep the tight tolerances: acc within 2e-3, NLLs within 5e-4.
    ref_hs = [x["calibrated"]["correct"] for x in v40hs]
    ref_lam = [x["calibrated"]["correct"] for x in v40lam]
    got_hs = [int(c) for c in fa["hs_correct"]]
    got_lam = [int(c) for c in fa["lam_correct"]]
    flip_hs = [j for j, (a, b) in enumerate(zip(got_hs, ref_hs)) if a != b]
    flip_lam = [j for j, (a, b) in enumerate(zip(got_lam, ref_lam)) if a != b]
    max_flip_margin = max([fa["hs_margin"][j] for j in flip_hs]) if flip_hs else 0.0
    print("qual A flips: HS %d/1000 (max margin %.4f) LAMBADA %d/1000" % (len(flip_hs), max_flip_margin, len(flip_lam)), flush=True)
    assert len(flip_hs) <= 2 and len(flip_lam) <= 2, "too many decision flips vs v40"
    assert max_flip_margin < 0.5, "flip on a wide-margin item: wiring suspect"
    for k, got, ref in (("hs_acc", fa["hs_acc"], v40cal["hs_acc"]), ("lambada_acc", fa["lambada_acc"], v40cal["lambada_acc"]),
                        ("lambada_nll", fa["lambada_nll"], v40cal["lambada_nll_per_tok"]), ("val_nll", fa["val_nll"], v40cal["val_nll"])):
        d = abs(got - ref)
        print("qual A %s: got=%.6f ref=%.6f |d|=%.2e %s" % (k, got, ref, d, "ok" if (d <= 2e-3) or (d <= t["val_nll"]) else "FAIL"), flush=True)
        assert (d <= 2e-3) if k.endswith("acc") else (d <= t["val_nll"]), "arm-A gate failed: " + k
    for d in doms:
        assert abs(fa["domains"][d] - v40cal["domains"][d]) <= t["domain_nll"], "arm-A domain gate failed: " + d
    print("GATE 1 PASSED: static arm A reproduces v40", flush=True)
    (r / "qualification-gate1.json").write_text(_json.dumps({"status": "passed", "armA": fa}, indent=2, default=str))
    return fa
def qualify_gate2(tok, model, inj, versions, ngram_fn, FZ, store, refdir):
    # Gate 2 (canonical gate): arm C alpha8 distributions + ordering C<A.
    import json as _json
    import arb_frozen
    import qualify
    r = proot()
    assert (r / "qualification-gate1.json").exists(), "gate1 first"
    fa = _json.loads((r / "qualification-gate1.json").read_text())["armA"]
    hs_rows, lam_rows, val_full, doms = _qual_inputs()
    arb, refdir = load_canonical_arb(r, inj, arb_frozen)
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON["alpha2_frozen"]
    try:
        fresh = qualify.qualify(model, hooks, store, ngram_fn, tok, val_full, doms, hs_rows, lam_rows, refdir)
    finally:
        hooks.close()
    assert fresh["val_nll"] <= fa["val_nll"], "ordering broken: C must beat static A on val"
    print("GATE 2 PASSED: arm C gate matches canonical + ordering C<A holds (%.5f<%.5f)" % (fresh["val_nll"], fa["val_nll"]), flush=True)
    (r / "qualification.json").write_text(_json.dumps({"status": "passed",
                                                       "armA": {"val_nll": fa["val_nll"], "hs_acc": fa["hs_acc"], "lambada_acc": fa["lambada_acc"]},
                                                       "armC": {"val_nll": fresh["val_nll"], "hs_acc": fresh["hs_acc"], "lambada_acc": fresh["lambada_acc"]}}, indent=2))
    return fresh
def qualify_stage(tok, model, inj, versions, ngram_fn, FZ, store, refdir):
    qualify_gate1(tok, model, inj, versions, ngram_fn, FZ, store)
    return qualify_gate2(tok, model, inj, versions, ngram_fn, FZ, store, refdir)
MILESTONE_BLOCKS = {500: 978, 750: 1464, 1000: 1956}  # 163/244/326 * 6; base 624
MILESTONE_FILES = {500: "arbitration-cont-500k.pt", 750: "arbitration-cont-750k.pt", 1000: "arbitration-cont-1000k.pt"}
def latest_milestone(r):
    import torch as _t
    for tag in (1000, 750, 500):
        mp = r / "milestones" / MILESTONE_FILES[tag]
        if mp.exists():
            return tag, _t.load(str(mp), map_location="cpu")
    return None, None
def train_legs(tok, model, inj, versions, ngram_fn, FZ, store, limit_blocks=None):
    import torch
    import arb_frozen
    import train as TR
    r = proot()
    assert (r / "qualification.json").exists(), "qualify first; training blocked"
    assert (r / "calib" / "calibration-continuation.json").exists(), "freeze first"
    arb0, _ = load_canonical_arb(r, inj, arb_frozen)
    mtag, mblob = latest_milestone(r)
    first_block = MILESTONE_BLOCKS[mtag] if mblob is not None else 624
    if limit_blocks is not None:
        assert limit_blocks > first_block, "leg target must extend past init"
    arb = arb_frozen.MemoryArbitration(hidden=1024).to("cuda")
    if mblob is not None:
        arb.w.data.copy_(mblob["w"].to("cuda"))
        arb.b.data.copy_(mblob["b"].to("cuda"))
        arb.alpha2_raw.data.copy_(arb0.alpha2_raw.detach().to("cuda"))
        init_src = "milestone-%dK-sha-verified" % mtag
    else:
        arb.w.data.copy_(arb0.w.detach().to("cuda"))
        arb.b.data.copy_(arb0.b.detach().to("cuda"))
        arb.alpha2_raw.data.copy_(arb0.alpha2_raw.detach().to("cuda"))
        init_src = "v49-canonical-weights"
    arb.alpha2_raw.requires_grad_(False)  # alpha2 frozen at canonical C value
    del arb0
    print("leg init: %s (fresh AdamW, per-leg re-warm; not an optimizer resume)" % init_src, flush=True)
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON["alpha2_frozen"]
    inj.set_memory(None)
    full, full_man = TR.ordered_full(r / "calib")
    ple_sig = json.loads((r / "compact" / "compact.json").read_text()) if (r / "compact" / "compact.json").exists() else {"mode": "full-ple"}
    try:
        res = TR.train(model, inj, hooks, arb, store, ngram_fn, full, full_man, T0, r / "resume", versions, ple_sig, limit_blocks, first_block)
    finally:
        hooks.close()
    cursor = res.get("cursor", full.shape[0])
    # Save exactly the milestone the leg ended on (never overwrite an earlier
    # milestone with later-leg weights; wall-stopped mid-leg saves nothing and resumes).
    blk_to_tag = {blk: tag for tag, blk in MILESTONE_BLOCKS.items()}
    if cursor in blk_to_tag:
        tag = blk_to_tag[cursor]
        if limit_blocks is None or cursor <= limit_blocks:
            mp = r / "milestones" / ("arbitration-cont-%dk.pt" % tag)
            mp.parent.mkdir(parents=True, exist_ok=True)
            payload = {"w": arb.w.detach().cpu(), "b": arb.b.detach().cpu(),
                       "alpha2_frozen": config.CANON["alpha2_frozen"],
                       "optimizer_resume_from_v49": False,
                       "init": init_src + "; fresh AdamW",
                       "schedule": "per-leg re-warm 0->1e-3/32 from leg start; cosine 1e-3->0 at leg end",
                       "cursor_block": cursor, "total_tokens": cursor * config.SEQ}
            msha, _ = TR.atomic_torch_save(payload, mp, ("w", "b"))
            print("milestone %dK checkpoint: %s sha=%s" % (tag, mp.name, msha[:16]), flush=True)
    return res
def eval_single(tok, model, inj, versions, ngram_fn, FZ, store, tag):
    # Evaluate exactly one milestone file (no re-eval of others).
    import torch
    import arb_frozen
    import evaluate
    import frozen_data
    r = proot()
    mp = r / "milestones" / MILESTONE_FILES[tag]
    assert mp.exists(), "missing " + mp.name
    val_full = frozen_data.load_blocks(r / "frozen-v1" / "tokens-full.uint32le", config.REVS["val_full_sha256"])
    doms = {d: frozen_data.load_blocks(r / "frozen-v1" / ("tokens-%s.uint32le" % d), config.REVS["domains"][d]) for d in frozen_data.DOMAINS}
    hs_rows, _ = frozen_data.hellaswag_rows(1000, need_token())
    lam_rows, _ = frozen_data.lambada_rows(1000, need_token())
    blob = torch.load(str(mp), map_location="cpu")
    arb = arb_frozen.MemoryArbitration(hidden=1024).to("cuda")
    arb.w.data.copy_(blob["w"].to("cuda"))
    arb.b.data.copy_(blob["b"].to("cuda"))
    arb.requires_grad_(False)
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON["alpha2_frozen"]
    try:
        res = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val_full, doms, hs_rows, lam_rows, T0, HARD_S)
    finally:
        hooks.close()
    rep = {"val_nll": res["val_nll"], "domains": res["domains"], "domain_mean": res["domain_mean"],
           "hs_acc": res["hs_acc"], "lambada_acc": res["lambada_acc"], "lambada_nll": res["lambada_nll"],
           "alpha8": res["alpha8"], "total_tokens": blob.get("total_tokens"),
           "hs_correct": res["hs_correct"], "lam_correct": res["lam_correct"], "lam_nlls": res["lam_nlls"],
           "val_blocks": res["val_blocks"], "dom_blocks": res["dom_blocks"]}
    (r / "milestones" / ("eval-%dk.json" % tag)).write_text(json.dumps(rep))
    print("%dK: val=%.5f dom=%.5f hs=%.4f lamb=%.4f/%.5f" % (tag, rep["val_nll"], rep["domain_mean"], rep["hs_acc"], rep["lambada_acc"], rep["lambada_nll"]), flush=True)
    return rep
def eval_milestones(tok, model, inj, versions, ngram_fn, FZ, store):
    import torch
    import arb_frozen
    import evaluate
    import frozen_data
    r = proot()
    val_full = frozen_data.load_blocks(r / "frozen-v1" / "tokens-full.uint32le", config.REVS["val_full_sha256"])
    doms = {d: frozen_data.load_blocks(r / "frozen-v1" / ("tokens-%s.uint32le" % d), config.REVS["domains"][d]) for d in frozen_data.DOMAINS}
    hs_rows, _ = frozen_data.hellaswag_rows(1000, need_token())
    lam_rows, _ = frozen_data.lambada_rows(1000, need_token())
    out = {}
    for tag in (500, 750, 1000):
        mp = r / "milestones" / ("arbitration-cont-%dk.pt" % tag)
        if not mp.exists():
            continue
        blob = torch.load(str(mp), map_location="cpu")
        arb = arb_frozen.MemoryArbitration(hidden=1024).to("cuda")
        arb.w.data.copy_(blob["w"].to("cuda"))
        arb.b.data.copy_(blob["b"].to("cuda"))
        arb.requires_grad_(False)
        hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
        hooks.fixed_alpha2 = config.CANON["alpha2_frozen"]
        try:
            res = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val_full, doms, hs_rows, lam_rows, T0, HARD_S)
        finally:
            hooks.close()
        rep = {"val_nll": res["val_nll"], "domains": res["domains"], "domain_mean": res["domain_mean"],
               "hs_acc": res["hs_acc"], "lambada_acc": res["lambada_acc"], "lambada_nll": res["lambada_nll"],
               "alpha8": res["alpha8"], "total_tokens": blob.get("total_tokens"),
               "hs_correct": res["hs_correct"], "lam_correct": res["lam_correct"], "lam_nlls": res["lam_nlls"],
               "val_blocks": res["val_blocks"], "dom_blocks": res["dom_blocks"]}
        (r / "milestones" / ("eval-%dk.json" % tag)).write_text(json.dumps(rep))
        out[tag] = rep
        print("%dK: val=%.5f dom=%.5f hs=%.4f lamb=%.4f/%.5f" % (tag, rep["val_nll"], rep["domain_mean"], rep["hs_acc"], rep["lambada_acc"], rep["lambada_nll"]), flush=True)
    return out
def eval_canonical_c(tok, model, inj, versions, ngram_fn, FZ, store):
    # Per-item eval of the canonical 319488 arm C (v49 weights) for paired contrasts.
    import torch
    import arb_frozen
    import evaluate
    import frozen_data
    r = proot()
    out = r / "canonical-c-eval.json"
    if out.exists():
        return json.loads(out.read_text())
    val_full = frozen_data.load_blocks(r / "frozen-v1" / "tokens-full.uint32le", config.REVS["val_full_sha256"])
    doms = {d: frozen_data.load_blocks(r / "frozen-v1" / ("tokens-%s.uint32le" % d), config.REVS["domains"][d]) for d in frozen_data.DOMAINS}
    hs_rows, _ = frozen_data.hellaswag_rows(1000, need_token())
    lam_rows, _ = frozen_data.lambada_rows(1000, need_token())
    arb, _ = load_canonical_arb(r, inj, arb_frozen)
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON["alpha2_frozen"]
    try:
        res = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val_full, doms, hs_rows, lam_rows, T0, HARD_S)
    finally:
        hooks.close()
    rep = {"val_nll": res["val_nll"], "domains": res["domains"], "domain_mean": res["domain_mean"],
           "hs_acc": res["hs_acc"], "lambada_acc": res["lambada_acc"], "lambada_nll": res["lambada_nll"],
           "alpha8": res["alpha8"], "total_tokens": config.CAL_BASE_TOTAL,
           "hs_correct": res["hs_correct"], "lam_correct": res["lam_correct"], "lam_nlls": res["lam_nlls"],
           "val_blocks": res["val_blocks"], "dom_blocks": res["dom_blocks"]}
    out.write_text(json.dumps(rep))
    print("canonical-C eval: val=%.5f dom=%.5f hs=%.4f lamb=%.4f/%.5f" % (
        rep["val_nll"], rep["domain_mean"], rep["hs_acc"], rep["lambada_acc"], rep["lambada_nll"]), flush=True)
    return rep
def run_contrasts():
    # Paired bootstrap CIs for the four staged contrasts. No GPU needed.
    import bootstrap
    r = proot()
    ec = json.loads((r / "canonical-c-eval.json").read_text())
    e5 = json.loads((r / "milestones" / "eval-500k.json").read_text())
    e7 = json.loads((r / "milestones" / "eval-750k.json").read_text())
    e1 = json.loads((r / "milestones" / "eval-1000k.json").read_text())
    cons = [bootstrap.contrast("500736-vs-canonical319488", e5, ec),
            bootstrap.contrast("749568-vs-500736", e7, e5),
            bootstrap.contrast("1001472-vs-749568", e1, e7),
            bootstrap.contrast("1001472-vs-canonical319488", e1, ec)]
    (r / "contrasts.json").write_text(json.dumps(cons, indent=2))
    for c in cons:
        print(bootstrap.summarize(c), flush=True)
    return cons
DECISION_MLP = {"arch": ["RMSNorm(h8)", "Linear(1024->32)", "SiLU", "Linear(32->1)", "sigmoid*0.5"],
                "status": "prepared-only; do not launch in this job"}
if __name__ == "__main__":
    print("use run.py --stage <inspect|freeze|qualify|train|eval>", flush=True)
