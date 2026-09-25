import json
import sys
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / 'reader-scale-20m'))
from canonical_cache import patch_cache, put, save, source

nb = json.loads((here.parent / 'reader-scale-20m' / 'kaggle_reader_20m.ipynb').read_text())
patch_cache(nb, 15000064, 16000000, 'compact-reader-r4-early')

trainer = source(nb, 'c-trainfn')
needle = '        seen=step*C.SEQ; iloss+=loss.item()*(C.SEQ-1); icount+=C.SEQ-1\n'
assert trainer.count(needle) == 1
put(nb, 'c-trainfn', trainer.replace(needle, needle +
    "        if step % 250 == 0: print('reader R4 progress', seen, flush=True)\n"))
put(nb, 'c-reader-scale', (here / 'r4_cell.py').read_text())
out = here / 'kaggle_reader_r4_early.ipynb'
save(nb, out)
print(out)
