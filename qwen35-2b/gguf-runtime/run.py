import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = HERE.parent / '.artifacts/gguf'
LLAMA = ROOT / 'llama.cpp'
BIN = LLAMA / 'build-qwengram-cpu/bin'
PLE = LLAMA / 'models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
PRECISIONS = {'bf16': 'BF16', 'q8': 'Q8_0', 'q4': 'Q4_K_M'}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def execute(args, log, env=None):
    start = time.time()
    with log.open('w') as f:
        subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                       env=env, cwd=ROOT, check=True)
    record = {'command': [str(a) for a in args], 'start_unix': start,
              'wall_seconds': time.time()-start, 'log_sha256': sha(log)}
    log.with_suffix('.run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(log.name, 'complete', round(record['wall_seconds'], 1), 's', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=('stock', 'qwengram'))
    a = p.parse_args()
    if a.arm == 'qwengram':
        definition = json.loads((HERE.parent / 'qwengram-2b.json').read_text())
        assert 'pending' not in definition['status'].lower(), 'Endpoint selection is pending'
    zip_path = ART / 'wikitext-2-raw-v1.zip'
    assert sha(zip_path) == 'ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11'
    text_path = ART / 'wiki.test.raw'
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith('/wiki.test.raw'))
        text_path.write_bytes(z.read(name))
    assert sha(text_path) == '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08'
    def model(precision):
        if a.arm == 'stock':
            return ART / ('stock-%s.gguf' % precision)
        return ART / 'release' / ('QwenGram-2B-%s.gguf' % precision)
    env = os.environ.copy()
    env.pop('QWENGRAM_PLE', None)
    if a.arm == 'qwengram':
        env['QWENGRAM_PLE'] = str(PLE)
    for key, precision in PRECISIONS.items():
        path = model(precision)
        if precision != 'BF16' and not path.exists():
            args = [BIN / 'llama-quantize', '--tensor-type', '^qwengram[.]=f32',
                    model('BF16'), path, precision, '8']
            execute(args, HERE / 'logs' / ('quant-%s-%s.log' % (a.arm, key)), env)
        log = HERE / 'logs' / ('%s-%s.log' % (a.arm, key))
        if log.exists() and log.with_suffix('.run.json').exists():
            print('Already completed', log.name, flush=True)
            continue
        args = [BIN / 'llama-perplexity', '-m', path, '-f', text_path,
                '-c', '256', '-b', '256', '-ub', '256', '--chunks', '64',
                '--ppl-output-type', '1', '--no-warmup', '-t', '8', '-ngl', '0']
        execute(args, log, env)
        rows = re.findall(r'^\s*\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s*$', log.read_text(), re.M)
        assert len(rows) == 64, (log, len(rows))


if __name__ == '__main__':
    main()
