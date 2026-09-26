import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LLAMA = ROOT / 'llama.cpp'
ART = HERE.parent / '.artifacts/gguf'
PLE = LLAMA / 'models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'


def run(label, model, backend='cpu', ngl=0, sidecar=PLE):
    env = os.environ.copy()
    env.pop('QWENGRAM_PLE', None)
    if sidecar:
        env['QWENGRAM_PLE'] = str(sidecar)
    args = [LLAMA / ('build-qwengram-%s/bin/llama-completion' % backend),
            '-m', model, '-p', 'The capital of France is', '-n', '8',
            '-no-cnv', '--temp', '0', '-s', '1234', '-ngl', str(ngl), '-c', '256',
            '-b', '256', '-ub', '256', '-t', '8', '--no-warmup']
    p = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=180)
    (HERE / 'logs' / ('smoke-' + label + '.log')).write_text(p.stdout + '\n' + p.stderr)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
            'command': [str(a) for a in args]}


def main():
    results = {}
    devices = subprocess.check_output([str(LLAMA / 'build-qwengram-vulkan/bin/llama-completion'),
                                       '--list-devices'], text=True)
    assert 'Vulkan0:' in devices, 'Vulkan device access unavailable'
    results['vulkan_devices'] = devices
    model = ART / 'release/QwenGram-2B-Q8_0.gguf'
    for label, sidecar, expected in (
            ('missing-ple', None, 'set QWENGRAM_PLE'),
            ('invalid-ple', ART / 'stock-BF16.gguf', 'PLE tensors missing')):
        r = run(label, model, sidecar=sidecar)
        assert r['returncode'] != 0 and expected in r['stderr'], r
        results[label] = {'rejected': True, 'expected_error': expected}
    cases = [('08b-regression', LLAMA / 'models/qwengram/QwenGram-0.8B-Q8_0.gguf', 'cpu', 0)]
    for precision in ('BF16', 'Q8_0', 'Q4_K_M'):
        path = ART / 'release' / ('QwenGram-2B-%s.gguf' % precision)
        cases += [(precision+'-cpu', path, 'cpu', 0),
                  (precision+'-vulkan', path, 'vulkan', 99)]
    for label, path, backend, ngl in cases:
        r = run(label, path, backend, ngl)
        assert r['returncode'] == 0, r
        results[label] = {'stdout': r['stdout'], 'command': r['command']}
        print(label, repr(r['stdout']), flush=True)
    (HERE / 'smoke.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
