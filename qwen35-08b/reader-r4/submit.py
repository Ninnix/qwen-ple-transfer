import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).parent
env = os.environ.copy()
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
kind = sys.argv[1]
assert kind in ('reader', 'arb')
meta = json.loads((here / ('kernel-metadata.json' if kind == 'reader' else 'arb-kernel-metadata.json')).read_text())
code = (here / meta['code_file']).read_text()
with tempfile.TemporaryDirectory(prefix='qwen-r4-submit-', dir='/tmp') as d:
    p = Path(d)
    (p / meta['code_file']).write_text(code)
    (p / 'kernel-metadata.json').write_text(json.dumps(meta))
    subprocess.run(['kaggle', 'kernels', 'push', '-p', str(p)], env=env, check=True)
