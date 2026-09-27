import ast
import hashlib
import json
import sys
from pathlib import Path

import torch

import build

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def defs(code):
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                 or isinstance(n, ast.Assign) and all(v.id.isupper() for t in n.targets
                                                     for v in ast.walk(t) if isinstance(v, ast.Name))]
    ns = {}
    exec(compile(tree, '<reference>', 'exec'), ns)
    return ns


def main():
    manifest = {}
    for path in sorted(HERE.glob('*/*.ipynb')):
        nb = load(path)
        cells = {c['id']: ''.join(c['source']) for c in nb['cells']}
        for name, code in cells.items():
            ast.parse('\n'.join(s for s in code.splitlines() if not s.startswith('%')), filename=name)
        for name in ('c-hashing', 'c-ple'):
            assert cells[name] == build.source(build.BASE, name)
        for name in ('c-reader', 'c-inject'):
            assert cells[name] == build.adapt(build.source(build.BASE, name))
        assert "SITES = (3, 11)" in cells['c-config']
        assert 'N_LAYERS: int = 32' in cells['c-config']
        assert 'HIDDEN: int = 2560' in cells['c-config']
        assert build.PROTOCOL['train_stream_sha256'] in cells['c-freeze-train']
        assert build.PIN['target_revision'] in cells['c-load']
        assert "device_map={'':0}" in cells['c-load']
        assert 'for dt in [torch.float16]' in cells['c-load']
        if build.CHECKPOINTED:
            assert "os.environ['CUDA_VISIBLE_DEVICES'] = '0'" in cells['c-env']
            assert 'torch.cuda.device_count() == 1' in cells['c-qualification']
        assert 'passed_before_optimizer_step' in cells['c-qualification']
        if build.MULTI or build.CHECKPOINTED:
            assert cells['c-execution-parity'] == build.execution_cell()
            assert list(cells).index('c-execution-parity') == list(cells).index('c-load') + 1
            assert 'reader_gradient_max_abs' in cells['c-execution-parity']
            assert repr(build.EXECUTION['strategy']) in cells['c-config']
        assert list(cells).index('c-qualification') > list(cells).index('c-load')
        if path.parent.name.startswith('arb-'):
            stage = int(path.parent.name.split('-')[1][:-1])
            assert 'reader-%dm-summary.json' % stage in cells['c-reader-load']
            assert 'reader-train-15m.json' in cells['c-reader-load']
            assert '((0, 978), (978, 1464))' in cells['c-train-arb']
            assert 'hidden=2560' in cells['c-arb-math']
            assert 'dec[3].register' in cells['c-arb-math']
            assert 'dec[11].register' in cells['c-arb-math']
            assert 'raw-%dm' % stage in cells['c-evaluation']
        meta = load(path.parent / 'kernel-metadata.json')
        assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['is_private'] == 'true'
        manifest[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(manifest) == 7

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    ref = defs(build.source(build.BASE, 'c-reader'))['SharedValueReader'](hidden=2560, gamma=.001)
    port = defs(build.adapt(build.source(build.BASE, 'c-reader')))['SharedValueReader'](gamma=.001)
    port.load_state_dict(ref.state_dict())
    h = torch.randn(2, 7, 2560, dtype=torch.float16)
    m = torch.randn(2, 7, 2560)
    a, b = ref(h, m), port(h, m)
    assert torch.equal(a, b)
    a.float().square().mean().backward(); b.float().square().mean().backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(ref.parameters(), port.parameters()))

    arb_ref = load(build.REF / 'arb-eval/arb_eval.ipynb')
    aclass = defs(build.source(arb_ref, 'c-arb-math'))['MemoryArbitration']
    bclass = defs(build.adapt(build.source(arb_ref, 'c-arb-math')))['MemoryArbitration']
    a, b = aclass(hidden=2560), bclass()
    with torch.no_grad(): a.w.normal_(0, .02)
    b.load_state_dict(a.state_dict())
    x = h.float().requires_grad_()
    av, bv = a.alpha8(x), b.alpha8(x)
    assert torch.equal(av, bv) and av.std() > 0
    assert torch.equal(a.alpha2(), b.alpha2())
    av.sum().backward(); bv.sum().backward()
    assert x.grad is None
    assert torch.equal(a.w.grad, b.w.grad) and torch.equal(a.b.grad, b.b.grad)

    from checkpointing import CheckpointedDecoder
    from types import SimpleNamespace
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(16, 32)
            self.down = torch.nn.Linear(32, 16)
        def forward(self, h):
            return h + self.down(torch.tanh(self.up(h)))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(Block() for _ in range(4))
    model.eval().requires_grad_(False)
    ns = defs(build.source(build.BASE, 'c-inject'))
    ns.update(C=SimpleNamespace(N_LAYERS=4),
              SharedValueReader=defs(build.source(build.BASE, 'c-reader'))['SharedValueReader'])
    inj = ns['ReaderInjection'](model, (0, 2), 32, 16, 1, .1)
    inj.set_memory(torch.randn(1, 7, 32))
    original = torch.randn(1, 7, 16)
    engine = CheckpointedDecoder(model.model.layers)
    counts = [0]*4
    def counter(site):
        def before(module, args): counts[site] += 1
        return before
    handles = [layer.register_forward_pre_hook(counter(i)) for i,layer in enumerate(model.model.layers)]
    outputs, gradients = [], []
    for enabled in (False, True):
        engine.enabled = enabled
        inj.zero_grad(set_to_none=True)
        counts[:] = [0]*4
        h = original
        for layer in model.model.layers: h = layer(h)
        h.square().mean().backward()
        outputs.append(h.detach())
        gradients.append([p.grad.clone() for p in inj.parameters()])
        assert counts == [1]*4, counts
    assert torch.equal(*outputs)
    assert all(torch.equal(x,y) for x,y in zip(*gradients))
    assert engine.body_calls > engine.checkpoint_calls == 4
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not model.training
    inj.close()
    for handle in handles: handle.remove()

    sys.path.insert(0, str(HERE.parent / 'qwen36-35b/src'))
    from qwen36_ple.hashing import ngram_indices
    ids = torch.randint(0, 248320, (4, 512))
    ids[:, [0, 1, 17, 255, 256, 511]] = 248044
    own = defs(build.source(build.BASE, 'c-hashing'))['ngram_indices']
    assert torch.equal(own(ids), ngram_indices(ids))
    report = {'reader_output_max_abs_diff': 0, 'reader_gradient_max_abs_diff': 0,
              'arbitration_output_max_abs_diff': 0, 'arbitration_gradient_max_abs_diff': 0,
              'gate_hidden_detached': True, 'addressing_positions': ids.numel(),
              'addressing_equal': True, 'hidden': 2560, 'sites': [3, 11],
              'checkpointed_decoder_cpu_output_and_gradient_equal':True,
              'checkpointed_injection_hooks_once':True,
              'notebook_shas': manifest}
    (HERE / 'results/local-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Verified 7 notebooks; reader/gate forward and gradients match; addressing exact on 2048 positions')


if __name__ == '__main__':
    main()
