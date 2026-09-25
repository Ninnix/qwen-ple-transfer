import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v16.log'
OUT = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v19sweep.txt'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
lines = []
for stream, t, data in msgs:
    if float(t) > 2900:
        txt = data.encode().decode('unicode_escape', errors='replace')
        if stream == 'stderr' and 'Warning' in txt:
            continue
        lines.append('[%s t=%s] %s' % (stream, t, txt[:500]))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'records after t=2900')
