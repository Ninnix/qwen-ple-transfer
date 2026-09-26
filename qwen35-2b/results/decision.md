# Original Qwengram-2B 15M transfer decision

**Endpoint update:** the [matched 10M/15M comparison](../milestone-compare/decision.md) retains REAL-10M + linear750 as the canonical balanced endpoint, with 15M as max-LM/research. The report below records the original 15M study.

The fixed transfer experiment completed: **15,000,064 reader tokens + 749,568
linear-arbitration tokens**, with IDX2/IDX8, R=1, and the backbone and PLE frozen.
No architecture sweep was run.

| Model | Full-val NLL | Perplexity | Reduction vs stock |
| --- | ---: | ---: | ---: |
| Stock Qwen3.5-2B | 2.649104 | 14.141361 | — |
| Raw REAL-15M | 2.607793 | 13.569075 | 4.047% |
| REAL-15M + linear750 | 2.611400 | 13.618098 | 3.700% |

1. **Does PLE transfer work at 2B? Yes.** The matched 500,224-token controls
   give REAL > PERMUTED > RANDOM > DISABLED on full validation. Every paired
   full-val comparison has a 95% interval excluding zero. REAL also beats
   PERMUTED and RANDOM in all five held-out domains.

2. **How much full-val improvement?** The requested final candidate reduces
   perplexity by **3.700%**, paired 95% CI **[3.563%, 3.839%]**. Its NLL delta
   is −0.037704 [−0.039149, −0.036279]. The raw reader achieves 4.047%
   [3.904%, 4.193%].

3. **Does the gain generalize? Yes, across all five measured domains.**
   General, code, math, scientific and multilingual NLL all improve over
   stock, with paired intervals below zero. Final five-domain mean NLL is
   **2.151835**, versus **2.180667** stock. HellaSwag accuracy rises from
   45.6% to 46.8%: +1.2 points [0.5, 2.0].

4. **Does dynamic arbitration remain useful? Yes, with a tradeoff.** Against
   raw REAL-15M, it improves LAMBADA NLL by −0.044776 [−0.050949, −0.038984]
   and accuracy by +1.4 points [0.4, 2.4]. It worsens full-val NLL by
   +0.003606 [0.003344, 0.003871] and slightly worsens every domain NLL.
   Final LAMBADA accuracy is 54.6% versus 54.1% stock; that stock comparison
   is unresolved at this sample size. Arbitration is useful for this
   benchmark balance, not a universal NLL improvement.

5. **How does it compare with 0.8B?** Final relative perplexity reduction is
   **3.700% versus 5.048%** for the canonical 0.8B recipe: 1.348 percentage
   points smaller, retaining about 73.3% of that relative gain.

6. **Which Kaggle execution strategy? Single T4 was sufficient and selected.**
   Actual reader training peaked at **7.784 GiB**, with roughly **942–997
   tokens/s** excluding validation. A split across two GPUs was unnecessary
   and was not benchmarked, so there is no measured two-GPU speed comparison.
   Successful kernels including qualification took **7.108 hours** total,
   excluding queue/upload and interruption-related idle time.

7. **Is a GGUF/runtime release justified? Yes, as an experimental release.**
   The controlled transfer signal, five-domain improvements and benchmark
   results justify implementation. Preserve the raw reader alongside the
   requested gated candidate and document the arbitration tradeoff. This
   experiment does not establish a universal task-accuracy gain or validate
   a future quantized runtime.

**Final identity:** [Qwengram-2B](../qwengram-2b-15m.json) = frozen Qwen3.5-2B +
REAL-15M R=1 two-site reader + linear750 dynamic later-site arbitration +
external Flash-Next PLE. The early learned alpha is 1.191673; full-val late
alpha mean/std are 0.113578/0.131596.

All 39 checkpoint/result-manifest entries were downloaded back from the
private dataset `ninnix/qwengram-2b-checkpoints` and matched their SHA256 and
size records. The [detailed metrics](metrics.md), [bootstrap data](arb-bootstrap.json),
[control comparisons](control-controls-500k-summary.json), [artifact hashes](milestone-shas.json),
and [persistence verification](persistence-verification.json) are preserved.
No further experiment was started after this report.
