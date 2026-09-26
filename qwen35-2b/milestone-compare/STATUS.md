# Milestone comparison complete

Kaggle version 2 completed. Version 1 failed on a manifest filename before training.
Only the missing 10M arbiter was trained (749,568 tokens). Stock reproduction was exact.
See decision.md and comparison.json for all four arms and paired confidence intervals.

Canonical balanced endpoint: REAL-10M + linear750, under the predeclared conservative accuracy rule.
15M clearly improves LM NLL but does not establish a benchmark accuracy gain.
The calibrated HellaSwag change -0.2 pp [-0.6,+0.2] is not a significant regression.
15M is preserved as the max-LM/research endpoint.

All 21 new or updated comparison artifacts were persisted to the private Kaggle checkpoint dataset,
downloaded again and SHA256/size verified. See persistence-verification.json.

The selected 10M endpoint is released as BF16, Q8_0 and Q4_K_M GGUFs at
https://huggingface.co/Ninnix96/Qwengram-2b.
All 73 published files were verified; see ../gguf-runtime/publication.json.
Matched runtime evaluation and PLE sidecar validation passed; see ../gguf-runtime/README.md.
The runtime and documentation were committed and pushed to Ninnix/llama.cpp master:
068fcb42662453bec15298ba6bd59f552190a468.
