import json
from pathlib import Path


def cell(nb, name):
    return next(c for c in nb['cells'] if c.get('id') == name)


def source(nb, name):
    return ''.join(cell(nb, name)['source'])


def put(nb, name, body):
    cell(nb, name)['source'] = body.splitlines(keepends=True)


def patch_cache(nb, start, end, name):
    assert start % 512 == 0 and end > start
    compact = source(nb, 'c-compact')
    old = '_train_toks=freeze_stream_tokens(tokenizer, _stream, 20000000)'
    assert compact.count(old) == 1
    compact = compact.replace(old, f'_train_toks=freeze_stream_tokens(tokenizer, _stream, {end})[{start}//C.SEQ:]')
    compact = compact.replace('compact-20m', name)
    needle = '    _val,_=load_validation(\'full\')\n'
    assert compact.count(needle) == 1
    compact = compact.replace(needle, "    _train_sha=hashlib.sha256(_train_toks.numpy().astype('<u4').tobytes()).hexdigest()\n" + needle)
    needle = "           'dataset_rev':_mfF['dataset_rev'],'train_skip_docs':_mfF['skip_docs'],\n"
    assert compact.count(needle) == 1
    compact = compact.replace(needle, needle + f"           'train_start':{start},'train_end':int({end}),'train_tokens_sha256':_train_sha,\n")
    needle = '    _cmp=CompactPLE(_cdir)\n'
    assert compact.count(needle) == 1
    compact = compact.replace(needle, needle + '''    def _file_sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1<<20),b''): h.update(block)
        return h.hexdigest()
    assert _cmp.meta==_meta
    assert (_cdir/'addrs.u32').stat().st_size==_n*4
    assert (_cdir/'rows.u8').stat().st_size==_n*C.ROW_DIM
    assert _file_sha(_cdir/'addrs.u32')==_meta['addrs_sha256']
    assert _file_sha(_cdir/'rows.u8')==_meta['rows_sha256']
    assert bool((_cmp.addrs[1:]>_cmp.addrs[:-1]).all())
    print('compact cache SHA/reopen verified', _meta['train_tokens_sha256'], flush=True)
''')
    put(nb, 'c-compact', compact)

    trainer = source(nb, 'c-trainfn')
    needle = '    if resume is not None and _skip: chunks=chunks[_skip:]\n'
    assert trainer.count(needle) == 1
    trainer = trainer.replace(needle, needle + '''    import hashlib
    assert store.meta['train_start']==_seen0 and store.meta['token_train']==chunks.numel()
    assert hashlib.sha256(chunks.numpy().astype('<u4').tobytes()).hexdigest()==store.meta['train_tokens_sha256']
    print('training suffix tokens SHA verified', store.meta['train_tokens_sha256'], flush=True)
''')
    put(nb, 'c-trainfn', trainer)


def save(nb, out):
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['execution_count'] = None
            c['outputs'] = []
    Path(out).write_text(json.dumps(nb, indent=1))
