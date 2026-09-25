import re

lines = open(r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v19sweep.txt', encoding='utf-8').read().splitlines()
found = []
for i, l in enumerate(lines):
    if '"gate": [' in l:
        a = lines[i + 1].strip().rstrip(',')
        b = lines[i + 2].strip().rstrip(',')
        # find enclosing checkpoint threshold by scanning back
        th = tok = None
        for j in range(i, max(0, i - 40), -1):
            if '"threshold":' in lines[j] and th is None:
                th = re.search(r'\d+', lines[j]).group(0)
            if '"tokens":' in lines[j] and tok is None:
                tok = re.search(r'\d+', lines[j]).group(0)
            if th and tok:
                break
        found.append((tok, th, a, b))
print(len(found), 'gate arrays')
seen = set()
for tok, th, a, b in found:
    key = (tok, th)
    if key in seen:
        continue
    seen.add(key)
    print('tokens', tok, 'th', th, 'gate', a, b)
