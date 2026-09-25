import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_build_out\qwen3-8-flash-next-ple-dataset-build.log'
OUT = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\build_v3.txt'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
lines = []
for stream, t, data in msgs:
    if 40 < float(t) < 58:
        lines.append('[%s t=%s] %s' % (stream, t, data.encode().decode('unicode_escape', errors='replace')[:350]))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'records in window')
