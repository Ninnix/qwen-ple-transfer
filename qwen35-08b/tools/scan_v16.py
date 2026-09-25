import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v16.log'
OUT = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\v16late.txt'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
lines = []
for stream, t, data in msgs:
    if float(t) > 300:
        lines.append('[%s t=%s] %s' % (stream, t, data.encode().decode('unicode_escape', errors='replace')[:450]))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'records after t=300')
