import json

D = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\tok'

def load(name):
    with open(D + '\\' + name, encoding='utf-8') as f:
        return json.load(f)

t = load('t35-08b.json')
s = load('flash.json')
tc = load('t35-08b-cfg.json')
sc = load('flash-cfg.json')

def vocab_map(tj):
    m = dict(tj['model']['vocab'])
    for a in tj.get('added_tokens', []):
        m[a['content']] = a['id']
    return m

tm, sm = vocab_map(t), vocab_map(s)
print('target entries:', len(tm), '| source entries:', len(sm))
print('target added_tokens:', len(t.get('added_tokens', [])), '| source added_tokens:', len(s.get('added_tokens', [])))
print('target model.vocab size:', len(t['model']['vocab']), '| source model.vocab size:', len(s['model']['vocab']))
print('target max id:', max(tm.values()), '| source max id:', max(sm.values()))

only_t = {k: tm[k] for k in tm if k not in sm}
only_s = {k: sm[k] for k in sm if k not in tm}
diff = {k: (tm[k], sm[k]) for k in tm if k in sm and tm[k] != sm[k]}
print('tokens only in target:', len(only_t), '| only in source:', len(only_s), '| same token diff id:', len(diff))
for k, v in list(only_t.items())[:15]:
    print('  T-only:', repr(k)[:60], v)
for k, v in list(only_s.items())[:15]:
    print('  S-only:', repr(k)[:60], v)
for k, v in list(diff.items())[:10]:
    print('  DIFF:', repr(k)[:60], v)

def specials(tj, cfg, tag):
    print('---', tag)
    for k in ('eos_token_id', 'bos_token_id', 'pad_token_id', 'unk_token_id', 'vocab_size',
              'image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'):
        v = cfg.get(k, cfg.get('text_config', {}).get(k, None)) if isinstance(cfg, dict) else None
        print('  cfg', k, '=', v)
    print('  tok eos/bos/pad/unk:',
          tj.get('eos_token_id') if 'eos_token_id' in tj else None,
          '| trie size:', len(tj.get('trie', {})) if isinstance(tj.get('trie'), dict) else tj.get('trie'))

specials(t, tc, 'target Qwen3.5-0.8B')
specials(s, sc, 'source Flash-Next-FP8')

inv_t = {v: k for k, v in tm.items()}
inv_s = {v: k for k, v in sm.items()}
for i in (248044, 248046, 248077, 248320 - 1):
    print('id', i, '-> target:', repr(inv_t.get(i))[:50], '| source:', repr(inv_s.get(i))[:50])

print('byte-identical tokenizer.json:', open(D + '\\t35-08b.json', 'rb').read() == open(D + '\\flash.json', 'rb').read())
