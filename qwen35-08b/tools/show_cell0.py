import json, re

P = r'C:\Users\nicolo.devangelista\PycharmProjects\qwen36-ple\qwen35-08b\kaggle_qwen35_08b_ple_reader.ipynb'
nb = json.loads(open(P, encoding='utf-8').read())
src = ''.join([x for x in nb['cells'] if x['id'] == 'c-env'][0]['source'])
red = re.sub(r"secret_value_[01] = .{0,80}", lambda m: m.group(0)[:20] + '<redacted>', src)
print(red[-700:])
body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('!', '%')))
compile(body, 'c-env', 'exec')
print('CELL0 COMPILES (magics stripped)')
