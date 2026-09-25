import json

import modal


app = modal.App("qwen36-ple-hash-test")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "transformers==5.16.1",
).add_local_dir("qwen36-35b/src", remote_path="/root/src")


@app.function(image=image, cpu=4, memory=8192, timeout=10 * 60)
def hash_test():
    import sys

    import torch
    from torch import nn
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    sys.path.insert(0, "/root/src")
    from qwen36_ple.hashing import head_layout, layer_multipliers, ngram_indices

    class CaptureEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty(0))

        def forward(self, indices):
            return indices.unsqueeze(-1)

    reference = Qwen4ExpTextNGramEmbedding.__new__(Qwen4ExpTextNGramEmbedding)
    nn.Module.__init__(reference)
    reference.layer_idx = 1
    reference.ngram_size = 3
    reference.context_len = 2
    reference.heads_per_ngram = 8
    reference.ngram_heads = 16
    reference.ple_layer_index = 0
    reference.unigram_vocab_size = 248320
    reference.ngram_vocab_size_base = 20_000_000
    reference.seed = 1234
    reference.eos_token_id = 248044
    sizes, offsets = head_layout()
    reference.register_buffer("layer_multipliers", torch.tensor(layer_multipliers(248320, 3)))
    reference.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes))
    reference.register_buffer("ngram_heads_offsets", torch.tensor(offsets))
    reference.ngram_embedding = CaptureEmbedding()

    tokens = torch.tensor(
        [[248044, 10, 20, 30, 248044, 40, 50], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    expected = reference(tokens, past_key_values=None)
    actual = ngram_indices(tokens)
    if not torch.equal(expected, actual):
        mismatch = torch.nonzero(expected != actual)[0].tolist()
        raise RuntimeError(f"Hash mismatch at {mismatch}: {expected[tuple(mismatch)]} != {actual[tuple(mismatch)]}")
    result = {
        "status": "passed",
        "multipliers": list(layer_multipliers(248320, 3)),
        "head_vocab_sizes": list(sizes),
        "head_offsets": list(offsets),
        "tokens": tokens.tolist(),
        "indices": actual.tolist(),
    }
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(hash_test.remote(), indent=2))
