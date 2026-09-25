import json, re

lines = open(r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v19sweep.txt', encoding='utf-8').read().splitlines()
CK = []
cur, depth = None, 0
for ln in lines:
    m = re.match(r'\[(\w+) t=([\d.]+)\] (.*)$', ln)
    if not m:
        continue
    txt = m.group(3)
    if txt.strip() == '{':
        cur, depth = {'t': float(m.group(2))}, 1
        continue
    if cur is None:
        continue
    if txt.strip() == '{':
        depth += 1
        continue
    if txt.strip() in ('},', '}'):
        depth -= 1
        if depth == 0:
            CK.append(cur)
            cur = None
        continue
    kv = re.match(r'"([^"]+)":\s*(.*?),?\s*$', txt.strip())
    if kv and depth == 1:
        k, v = kv.group(1), kv.group(2).rstrip(',')
        try:
            cur[k] = json.loads(v)
        except Exception:
            cur[k] = v
    gm = re.match(r'"readers\.([\d]+)\.gamma":\s*([0-9.e+-]+)', txt.strip())
    if gm:
        cur.setdefault('gammas', {})[gm.group(1)] = float(gm.group(2))
    if txt.strip().startswith('0.') and 'gate' in str(cur.get('gate', '')) or re.match(r'^[0-9.e+-]+,$', txt.strip()) and cur is not None and depth == 1:
        if 'gate' not in cur or not isinstance(cur.get('gate'), list):
            pass
print('checkpoints parsed:', len(CK))
for c in CK:
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items() if k in (
        't', 'tokens', 'threshold', 'val_set', 'train_loss', 'val_loss', 'dval', 'ppl',
        'gnorm', 'w_k_update', 'w_v_update', 'beta_update', 'tok_s', 'peak_GiB', 'rss_MiB')},
          'gammas:', c.get('gammas'), 'gate:', c.get('gate'))
