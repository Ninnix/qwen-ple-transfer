import json
from pathlib import Path

here = Path(__file__).parent
nb = json.loads((here / 'kaggle_reader_20m.ipynb').read_text())

def cell(name):
    return next(c for c in nb['cells'] if c.get('id') == name)

compact = ''.join(cell('c-compact')['source'])
old = '_train_toks=freeze_stream_tokens(tokenizer, _stream, 20000000)'
assert compact.count(old) == 1
compact = compact.replace(old, old + '[15000064//512:]')
compact = compact.replace('compact-20m', 'compact-20m-suffix')
cell('c-compact')['source'] = compact.splitlines(keepends=True)

trainer = ''.join(cell('c-trainfn')['source'])
old = '        seen=step*C.SEQ; iloss+=loss.item()*(C.SEQ-1); icount+=C.SEQ-1\n'
assert trainer.count(old) == 1
trainer = trainer.replace(old, old + "        if step % 500 == 0: print('reader20 progress', seen, flush=True)\n")
cell('c-trainfn')['source'] = trainer.splitlines(keepends=True)

scale = ''.join(cell('c-reader-scale')['source'])
old = "assert meta['token_train'] == 20000256 and meta['token_val'] == 524288"
assert scale.count(old) == 1
scale = scale.replace(old, "assert meta['token_train'] == 5000192 and meta['token_val'] == 524288")
cell('c-reader-scale')['source'] = scale.splitlines(keepends=True)

out = here / 'kaggle_reader_20m_suffix.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(out)
