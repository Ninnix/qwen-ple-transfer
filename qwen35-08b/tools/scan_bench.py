import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v16.log'
OUT = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v17bench.txt'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
want = ('tok/s', 'speedup', 'COMPACT', 'compact PLE', 'equivalence', 'BENCH',
        'PROFILE', 'stage ', 'dedup', 'requested addresses', 'cache dir',
        'REAL_SMOKE', 'Traceback', 'Error')
lines = []
for stream, t, data in msgs:
    txt = data.encode().decode('unicode_escape', errors='replace')
    if any(k in txt for k in want):
        lines.append('[%s t=%s] %s' % (stream, t, txt[:400]))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'kept')
