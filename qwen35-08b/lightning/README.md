# Lightning continuation (T4): linear IDX8 arbiter 319K -> 1M
Canonical v1 = Kaggle arm C (REAL-5M R=1, alpha2=1.267012596130371, learned linear IDX8 gate).
Kaggle stack untouched; all changes here are infrastructure-only (paths, persistence, resume).
Frozen reuse: `sync_frozen.py` extracts c-hashing/c-reader/c-inject/c-ple verbatim; `arb_frozen.py`
pins the gate math; scoring/data builders mirror the arb cell with only the root path moved.
Run order on Studio (free T4, ~4h restarts; legs stop cleanly at 3h30):
1. Set `HF_TOKEN` in the Studio environment.
2. `python sync_frozen.py` then `python run.py --stage inspect` (disk report + full/compact plan).
3. Copy Kaggle canonicals to persistent `canonical-v1/` (arbitration-c.pt/json, alpha8-stats.json,
   summary.json) + base `ckpts/real-5m-r1.reader.safetensors` (SHA-verified) + frozen eval blocks.
4. `python run.py --stage freeze` (calibration 500736/749568/1001472 manifests + compact cache SHAs).
5. `python run.py --stage qualify` (must PASS; aborts training on material drift).
6. `python run.py --stage train` (rerun after each restart; resumes from last verified checkpoint).
7. `python run.py --stage eval` (frozen full-val + HS/LAMBADA + 5 domains at 500K/750K/1M).
Storage: full 48.7 GiB PLE only if persistent free >= 70 GiB; else persistent compact cache
(sequential shard fetch, deduped global addresses, sampled equivalence vs master).
Checkpoints hold w/b + optimizer + scheduler + token count + step + cursors + RNGs + manifest SHAs +
base reader SHA + PLE/cache SHA. Milestones saved independently at 500K/750K/1M.
Decision: gains through 1M keep linear; saturate/reverse stops. MLP (RMSNorm/Linear1024-32/SiLU/
Linear32-1/sigmoid*0.5) is prepared only, never launched here.
