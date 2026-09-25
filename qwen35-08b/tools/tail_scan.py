import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_run_out\qwen3-5-0-8b-ple-target-side-reader-p100.log'
OUT = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\tail.txt'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
lines = []
for stream, t, data in msgs:
    if float(t) > 264.09 and stream == 'stdout':
        lines.append('[t=%s] %s' % (t, data.encode().decode('unicode_escape', errors='replace')[:300]))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'records written')
