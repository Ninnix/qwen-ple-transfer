# Frozen scoring, same definitions as Kaggle arb cell. Logprob only, no generation.
# HS: argmax over option logprob sums. LAMBADA: teacher-forced last-word NLL + all-tokens-correct acc.
# Block NLL: cross_entropy sum over non-shifted tokens divided by token count.
import time
import torch
import torch.nn.functional as F
def alpha_stats(t):
    g = t.detach().float().reshape(-1)
    q = torch.quantile(g, torch.tensor([0.05, 0.5, 0.95]))
    return {"mean": float(g.mean()), "std": float(g.std(correction=0)),
            "p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "n": int(g.numel())}
def block_nll(model, hooks, store, ngram_fn, blocks, bucket, collect, t0, hard_s):
    out = []
    with torch.inference_mode():
        for bi in range(blocks.shape[0]):
            if (time.perf_counter() - t0) > hard_s:
                break
            b = blocks[bi].unsqueeze(0)
            hooks.set_memory(store.lookup(ngram_fn(b.cpu())).to("cuda"))
            lg = model(input_ids=b.to("cuda"), use_cache=False).logits
            collect[bucket].append(hooks.last_alpha8.reshape(-1).cpu())
            s = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]),
                                b.to("cuda")[:, 1:].reshape(-1), reduction="sum").item()
            out.append({"sum": s, "n": int(b[:, 1:].numel())})
    mean = sum(x["sum"] for x in out) / max(1, sum(x["n"] for x in out))
    return out, mean
def logprob_opts(model, hooks, store, ngram_fn, tok, prompt, conts, bucket, collect):
    p = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    outs = []
    with torch.inference_mode():
        for c in conts:
            t = tok(c if c.startswith(" ") else " " + c, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = torch.cat([p, t], dim=1)
            hooks.set_memory(store.lookup(ngram_fn(ids)).to("cuda"))
            lg = model(input_ids=ids.to("cuda"), use_cache=False).logits.float()
            collect[bucket].append(hooks.last_alpha8.reshape(-1).cpu())
            lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
            outs.append(lp.gather(1, t[0].to("cuda").unsqueeze(1)).sum().item())
    return outs
def lambada_score(model, hooks, store, ngram_fn, tok, context, target):
    p = tok(context, return_tensors="pt", add_special_tokens=False)["input_ids"]
    t = tok(target if target.startswith(" ") else " " + target, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = torch.cat([p, t], dim=1)
    with torch.inference_mode():
        hooks.set_memory(store.lookup(ngram_fn(ids)).to("cuda"))
        lg = model(input_ids=ids.to("cuda"), use_cache=False).logits.float()
        lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
        nll = -lp.gather(1, t[0].to("cuda").unsqueeze(1)).sum().item()
        acc = int(bool((lp.argmax(-1).cpu() == t[0]).all()))
    return nll, acc, int(t.shape[1])
def eval_all_static(model, inj, store, ngram_fn, tok, val_full, dom_blocks, hs_rows, lam_rows, t0, hard_s):
    # Static arm (e.g. arm A via in-memory gamma scaling): plain injection, no gate.
    import time as _t
    hs_c = []
    hs_margin = []
    with torch.inference_mode():
        for r in hs_rows:
            if (_t.perf_counter() - t0) > hard_s:
                break
            p = tok(r["ctx"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            lps = []
            for c in r["endings"]:
                t = tok(c if c.startswith(" ") else " " + c, return_tensors="pt", add_special_tokens=False)["input_ids"]
                ids = torch.cat([p, t], dim=1)
                inj.set_memory(store.lookup(ngram_fn(ids)).to("cuda"))
                lg = model(input_ids=ids.to("cuda"), use_cache=False).logits.float()
                lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
                lps.append(lp.gather(1, t[0].to("cuda").unsqueeze(1)).sum().item())
            hs_c.append(int(max(range(len(r["endings"])), key=lambda i: lps[i]) == r["label"]))
            srt = sorted(lps, reverse=True)
            hs_margin.append(srt[0] - srt[1] if len(srt) > 1 else 0.0)
        lam_c, lam_n = [], []
        for r in lam_rows:
            if (_t.perf_counter() - t0) > hard_s:
                break
            p = tok(r["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            t = tok(r["target"] if r["target"].startswith(" ") else " " + r["target"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = torch.cat([p, t], dim=1)
            inj.set_memory(store.lookup(ngram_fn(ids)).to("cuda"))
            lg = model(input_ids=ids.to("cuda"), use_cache=False).logits.float()
            lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
            nll = -lp.gather(1, t[0].to("cuda").unsqueeze(1)).sum().item()
            lam_c.append(int(bool((lp.argmax(-1).cpu() == t[0]).all())))
            lam_n.append(nll / max(1, int(t.shape[1])))
        def _blk(blocks):
            out = []
            for bi in range(blocks.shape[0]):
                if (_t.perf_counter() - t0) > hard_s:
                    break
                b = blocks[bi].unsqueeze(0)
                inj.set_memory(store.lookup(ngram_fn(b.cpu())).to("cuda"))
                lg = model(input_ids=b.to("cuda"), use_cache=False).logits
                s = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), b.to("cuda")[:, 1:].reshape(-1), reduction="sum").item()
                out.append({"sum": s, "n": int(b[:, 1:].numel())})
            return out, sum(x["sum"] for x in out) / max(1, sum(x["n"] for x in out))
        vb, v = _blk(val_full)
        doms = {}
        dom_detail = {}
        for d, b in dom_blocks.items():
            _db, _dd = _blk(b)
            doms[d] = _dd
            dom_detail[d] = _db
    inj.set_memory(None)
    return {"hs_acc": sum(hs_c) / max(1, len(hs_c)), "hs_correct": hs_c, "hs_margin": hs_margin,
            "lambada_acc": sum(lam_c) / max(1, len(lam_c)), "lambada_nll": sum(lam_n) / max(1, len(lam_n)),
            "lam_correct": lam_c, "lam_nlls": lam_n,
            "val_nll": v, "val_blocks": vb, "domains": doms, "dom_blocks": dom_detail,
            "domain_mean": sum(doms.values()) / max(1, len(doms))}
def eval_all(model, hooks, store, ngram_fn, tok, val_full, blocks_in, hs_rows, lam_rows, t0, hard_s):
    from collections import defaultdict
    collect = defaultdict(list)
    hs_c = []
    for r in hs_rows:
        lps = logprob_opts(model, hooks, store, ngram_fn, tok, r["ctx"], r["endings"], "hellaswag", collect)
        hs_c.append(int(max(range(len(r["endings"])), key=lambda i: lps[i]) == r["label"]))
    lam_c, lam_n = [], []
    for r in lam_rows:
        nll, acc, nt = lambada_score(model, hooks, store, ngram_fn, tok, r["context"], r["target"])
        collect["lambada"].append(hooks.last_alpha8.reshape(-1).cpu())
        lam_c.append(acc)
        lam_n.append(nll / max(1, nt))
    vb, v = block_nll(model, hooks, store, ngram_fn, val_full, "full-val", collect, t0, hard_s)
    doms = {}
    dom_detail = {}
    for d in blocks_in:
        _db, _dd = block_nll(model, hooks, store, ngram_fn, blocks_in[d], d, collect, t0, hard_s)
        doms[d] = _dd
        dom_detail[d] = _db
    stats = {k: alpha_stats(torch.cat(vs)) for k, vs in collect.items() if vs}
    return {"hs_acc": sum(hs_c) / max(1, len(hs_c)), "hs_correct": hs_c,
            "lambada_acc": sum(lam_c) / max(1, len(lam_c)), "lambada_nll": sum(lam_n) / max(1, len(lam_n)),
            "lam_correct": lam_c, "lam_nlls": lam_n,
            "val_nll": v, "val_blocks": vb, "domains": doms, "dom_blocks": dom_detail,
            "domain_mean": sum(doms.values()) / max(1, len(doms)), "alpha8": stats}
