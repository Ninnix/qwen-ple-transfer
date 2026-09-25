# REAL R=4 early reader ablation

**Decision: stop at +999,936 reader tokens.** REAL-15M R=1 with its linear750 gate remains the balanced canonical. REAL-20M R=1 with its linear750 gate remains the max-LM endpoint. The R=4 reader and its separately calibrated linear750 gate are preserved as a research endpoint. No +2.5M R=4 continuation was run.

The R=4 reader started from the SHA-verified 15M R=1 reader. Both shared value projections and both gammas were copied exactly. Each set of four keys was copied from the R=1 key with paired positive and negative perturbations at 1e-4 relative scale; beta was repeated. Initial R=4 versus R=1 reader output differed by at most 2.38e-7 at each site. On one frozen model block, the maximum downstream fp16 logit difference was 0.0390625. Backbone and PLE stayed frozen. The expanded reader used fresh AdamW state, the established continuation LR shape, and the same FineWebEdu stream. The exact 999,936-token suffix SHA-256 was `0178b3c726deef4caf302b86f18921fa259d744c4c690b2f72da9733f52b33c5`.

The [canonical suffix-cache path](../../reader-scale-20m/CACHE_IO.md) built a 2.003 GiB compact cache for the new training suffix plus frozen full validation. Reopened address and row files matched their SHA-256 records; 2,048 sampled lookups matched the immutable PLE source exactly. The reader weights were saved, reopened, and SHA-verified **before** full validation. The bundled full optimizer resume was reopened and SHA-verified afterward.

## Frozen evaluation against exact 15M canonical

The R=4 checkpoint was given a fresh linear IDX8 gate trained on the established balanced 500,736 → 749,568-token calibration prefix. Alpha2 remained fixed at 1.267012596. Both gate checkpoints were saved and reopened before evaluation. The 15M baseline is the exact SHA-verified canonical evaluation artifact. Negative NLL differences favor R=4. Intervals use 10,000 paired block or example bootstrap resamples.

| Metric | 15M R=1 + linear750 | 16M R=4 + linear750 | R=4 minus R=1 [95% CI] |
| --- | ---: | ---: | ---: |
| Full-val NLL | 2.853786 | 2.852958 | -0.000828 [-0.000889, -0.000765] |
| Five-domain mean NLL | 2.384680 | 2.384245 | -0.000435 [-0.000539, -0.000326] |
| LAMBADA-1000 NLL | 2.217277 | 2.216757 | -0.000519 [-0.001525, +0.000493] |
| HellaSwag-1000 accuracy | 0.400 | 0.399 | -0.001 [-0.003, 0.000] |
| LAMBADA-1000 accuracy | 0.456 | 0.455 | -0.001 [-0.005, +0.002] |

| Domain NLL | 15M R=1 | 16M R=4 | R=4 minus R=1 [95% CI] |
| --- | ---: | ---: | ---: |
| General | 3.134094 | 3.132702 | -0.001392 [-0.001651, -0.001128] |
| Code | 1.495201 | 1.495545 | +0.000345 [+0.000117, +0.000574] |
| Math | 1.402346 | 1.402984 | +0.000638 [+0.000391, +0.000880] |
| Scientific | 2.244491 | 2.243079 | -0.001412 [-0.001626, -0.001202] |
| Multilingual | 3.647268 | 3.646916 | -0.000353 [-0.000589, -0.000114] |

Full-val and equal-weight domain mean NLL improve with paired intervals below zero. General, scientific, and multilingual improve. **Code and math regress with paired intervals above zero.** LAMBADA NLL and both benchmark accuracy changes are unresolved. The experiment adds reader tokens and branches together; these mixed held-out results do not establish a balanced capacity benefit beyond 15M R=1. The existing 20M R=1 endpoint also remains better on full-val, domain-mean, and LAMBADA NLL: paired R=4 minus 20M differences are +0.001421 [+0.001288, +0.001553], +0.000887 [+0.000641, +0.001132], and +0.005504 [+0.003596, +0.007324]. That comparison is context because reader training tokens differ.

## Alpha8 distributions

| Frozen set | 15M mean | R=4 mean | R=4 std | R=4 p05 | R=4 p50 | R=4 p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-val | 0.124661 | 0.124384 | 0.143500 | 0.006786 | 0.058115 | 0.469835 |
| HellaSwag | 0.171142 | 0.169519 | 0.165916 | 0.010007 | 0.095127 | 0.488408 |
| LAMBADA | 0.162809 | 0.161442 | 0.154377 | 0.010431 | 0.097312 | 0.479219 |
| General | 0.133051 | 0.133061 | 0.145836 | 0.009404 | 0.066317 | 0.484335 |
| Code | 0.117820 | 0.118792 | 0.133325 | 0.007846 | 0.057633 | 0.435968 |
| Math | 0.128265 | 0.128281 | 0.150981 | 0.008231 | 0.055118 | 0.476641 |
| Scientific | 0.128738 | 0.128589 | 0.153386 | 0.005804 | 0.053604 | 0.485508 |
| Multilingual | 0.124943 | 0.124710 | 0.143655 | 0.005354 | 0.059116 | 0.468350 |

Paired per-block or per-example alpha8 distribution changes are in [alpha8-paired.json](reader-arb-r4/alpha8-paired.json). On full-val, paired changes for mean, std, p05, p50, p95 are -0.000277 [-0.000339, -0.000214], -0.001352 [-0.001388, -0.001317], +0.000268 [+0.000246, +0.000289], +0.000892 [+0.000783, +0.001006], and -0.002332 [-0.002471, -0.002191]. On LAMBADA they are -0.001398 [-0.001501, -0.001294], -0.002321 [-0.002388, -0.002257], +0.000180 [+0.000120, +0.000241], +0.000807 [+0.000473, +0.001158], and -0.004237 [-0.004447, -0.004035]. These paired item means differ from token-weighted global differences where examples have different scored lengths.

## Checkpoints and provenance

| Artifact | SHA-256 |
| --- | --- |
| 15M R=1 parent reader | `e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc` |
| R=4 pre-evaluation reader weights | `cd3a716c24a1f38d55180749192fa9fae761bfff5cfa45d29ff2c16f4167536b` |
| R=4 raw optimizer checkpoint | `4ee664e3d91c9b135abeae14edb862dd5bc4dd4282a2997bcf6d43d3722b176a` |
| R=4 bundled full optimizer resume | `3932457db2f0e4da88b6a0ce3adc8da36f0316083c3250c72ab659e34778c793` |
| R=4 linear gate, 500,736 tokens | `03d30265f41f18bc215b135a7efc31320cf73d62f19101b3753438462a8f7b9d` |
| R=4 linear gate, 749,568 tokens | `53a71cc77b8c1c0b539f1dd838b368f7bd48365031cff0d8b093e2a5c7505ac1` |

Reader kernel: `ninnix/qwen-ple-real-r4-reader-warm-start-early-1m-t4/3`. Reader dataset: `ninnix/qwen-ple-reader-r4-early-1m-milestone`. Gate/evaluation kernel: `ninnix/qwen-ple-real-r4-early-arbitration-t4/1`. Result dataset: `ninnix/qwen-ple-real-r4-early-arbitration-results`. The [reader](reader-r4-shas.json) and [gate](arb-r4-shas.json) manifests pin private dataset contents. [Summary](reader-arb-r4/summary.json), [paired contrasts](reader-arb-r4/contrast.json), and [R=4 versus 20M context](contrast-r4-vs20m-context.json) retain every per-block or per-example comparison.
