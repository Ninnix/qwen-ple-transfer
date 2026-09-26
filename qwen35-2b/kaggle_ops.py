import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
key = os.environ.get('KAGGLE_API_TOKEN')
if not key:
    raise SystemExit('Set KAGGLE_API_TOKEN before running Kaggle operations')
env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle-config'
p = subprocess.Popen([str(ROOT / '.venv/bin/kaggle'), *sys.argv[1:]],
                     cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
for line in p.stdout:
    print(line.replace(key, '[REDACTED]'), end='', flush=True)
raise SystemExit(p.wait())
