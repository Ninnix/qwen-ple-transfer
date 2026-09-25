# Lightning continuation config. Pins only; no training logic.
# Canonical v1 = Kaggle Stage A2-arb arm C (frozen REAL-5M R=1 IDX 2+8 + learned linear IDX8 gate).
# Kaggle stack is reference and stays untouched; this file is infrastructure-only.
CANON = {
    "base_reader_file": "real-5m-r1.reader.safetensors",
    "base_reader_run": "real-5m-r1.run.json",
    "base_reader_sha256": "078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594",
    "base_reader_tokens": 5000192,
    "gamma2": 0.06166713312268257,
    "gamma8": 0.07537073642015457,
    "placement_idx": [2, 8],
    "branches": 1,
    "alpha2_frozen": 1.267012596130371,  # learned C value; frozen for this job
    "arb_dataset": "ninnix/qwen-ple-a2-arb",  # provides arbitration-c.pt/json (w,b)
    "ckpt_dataset": "ninnix/qwen-ple-reader-checkpoints",
    "v34_dataset": "ninnix/qwen-ple-a2-v34",
    "v40_dataset": "ninnix/qwen-ple-a2-final-v40",
    "v40_summary_sha256": "dd584526893d09956a600173286a64fc5c96dda5afe129463fc34bc5dfbebec1",
}
ADDR = {"vocab": 248320, "eos": 248044, "seed": 1234, "ngram": 3,
        "heads_per_ngram": 8, "rows_per_part": 2500012, "vocab_base": 20000000}
REVS = {
    "target_id": "Qwen/Qwen3.5-0.8B",
    "source_id": "Qwen/Qwen3.8-Flash-Next-FP8",
    "tokenizer_rev": "2fc06364715b967f1860aea9cf38778875588b17",
    "ple_revision": "236dfdf285828023ca3bcd3f37366c58a3469b13",
    "source_rev_prefix": "236dfdf28582",
    "hellaswag": {"dataset": "Rowan/hellaswag", "rev": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "n": 1000},
    "lambada": {"dataset": "EleutherAI/lambada_openai", "rev": "900124bf3b8235c6daf21033af9948b3f07346c4", "n": 1000},
    "val_full_sha256": "26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c",
    "domains": {
        "general": "a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5",
        "code": "735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421",
        "math": "c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648",
        "scientific": "c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8",
        "multilingual": "65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55",
    },
}
# Calibration: balanced 6-way, block-aligned (SEQ=512). Base 104 blocks/domain.
# Continuation extends streams, never replays base. Totals are exact multiples of 6*512.
SEQ = 512
CAL_SOURCES = ("general", "narrative", "code", "math", "scientific", "multilingual")
CAL_BASE_PER_DOMAIN = 53248   # 104 blocks
CAL_BASE_TOTAL = 319488       # 624 blocks
HELDOUT_PREFIX = 32768        # skipped per overlapping source, preserved on extend
MILESTONES = {  # per-domain blocks -> exact totals
    500: {"per_domain_blocks": 163, "per_domain_tokens": 83456, "total": 500736},
    750: {"per_domain_blocks": 244, "per_domain_tokens": 124928, "total": 749568},
    1000: {"per_domain_blocks": 326, "per_domain_tokens": 166912, "total": 1001472},
}
CAL_SEED = 1234
# Arbitration hyperparams, identical to Kaggle arb cell; scheduler is explicit continuation.
ARB = {"lr_peak": 1e-3, "wd": 0.0, "warm_steps": 32, "lambda_mean_reg": 2.0,
       "alpha8_target": 0.125, "hidden": 1024, "params": 1026,
       "b_init": -1.0986122886681098}
# Qualification tolerances: tight but not bitwise (T4 vs P100, CUDA kernels differ).
# Diagnosed 2026-09-15: manifests align 1000/1000, single HS flip at margin 0.068
# (acc 0.4050 vs 0.4060) from fp16 kernel noise. Per-example exactness is therefore
# replaced by flip-count + margin-closeness; aggregates keep tight tolerances.
QUAL_TOL = {"val_nll": 5e-4, "domain_nll": 5e-4, "lambada_nll": 5e-4,
            "acc_tol": 2e-3, "max_flips": 2, "flip_margin": 0.5,
            "alpha8_mean": 5e-3, "alpha8_std": 5e-3,
            "alpha8_q": 1e-2, "logit_max": 2.5e-1, "probe_nll": 5e-3}
# Canonical arm-C alpha8 reference (full-val) for the qualification gate.
ALPHA8_REF = {"full-val": {"mean": 0.12246814370155334, "std": 0.1301114559173584,
                           "p05": 0.01889512501657009, "p50": 0.0652507022023201,
                           "p95": 0.45652467012405396}}
# Studio restart handling.
WALL_LIMIT_S = 3 * 3600 + 30 * 60  # 3h30 hard stop, well before ~4h restart
PERSISTENT_ENV = "LIGHTNING_PERSISTENT_DIR"  # Studio persistent mount; default below
DEFAULT_PERSISTENT = "/teamspace/studios/this_studio/persistent"
SCRATCH_ENV = "LIGHTNING_SCRATCH_DIR"
