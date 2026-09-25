import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'ple-dataset'
OUT.mkdir(parents=True, exist_ok=True)

# Reuse index via HF hub API through Invoke-RestMethod output cached below by caller;
# here we rebuild the parts map deterministically: shard_i lives in the file mapped
# in model.safetensors.index.json. We fetch the index with plain urllib + token.
import urllib.request
tok = os.environ.get('HF_TOKEN')
assert tok, 'set HF_TOKEN'
req = urllib.request.Request(
    'https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/236dfdf285828023ca3bcd3f37366c58a3469b13/model.safetensors.index.json',
    headers={'Authorization': 'Bearer ' + tok})
idx = json.loads(urllib.request.urlopen(req, timeout=120).read().decode('utf-8'))
wm = idx['weight_map']
parts = {}
for i in range(128):
    key = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight' % i
    parts[str(i)] = wm[key]
files = sorted(set(parts.values()))
manifest = {
    'format': 'qwen-flashnext-ple-fp8',
    'version': 1,
    'ple_revision': '236dfdf285828023ca3bcd3f37366c58a3469b13',
    'source_model': 'Qwen/Qwen3.8-Flash-Next-FP8',
    'rows_per_part': 2500012,
    'row_dim': 160,
    'parts': parts,
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(OUT / 'FILES.txt').write_text('\n'.join(files) + '\n')
print('parts:', len(parts), '| files:', len(files))
print('first:', files[0], '| last:', files[-1])
