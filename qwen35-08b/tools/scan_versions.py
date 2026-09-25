import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_run_out\qwen3-5-0-8b-ple-target-side-reader-p100.log'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
for stream, t, data in msgs:
    data = data.encode().decode('unicode_escape', errors='replace')
    if 'torch' in data and ('cap' in data or '2.' in data) or 'compute_cap' in data or re.match(r'^[\d.]+ [\d.]+', data.strip()):
        print('[%s t=%s] %s' % (stream, t, data[:400]))
