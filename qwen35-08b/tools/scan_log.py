import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_run_out\qwen3-5-0-8b-ple-target-side-reader-p100.log'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
print('total records:', len(msgs))
for stream, t, data in msgs:
    data = data.encode().decode('unicode_escape', errors='replace')
    if 'Traceback' in data or 'Error' in data or 'error' in data or 'assert' in data or stream == 'stdout' and ('ok' in data or 'OK' in data or ' carnival' in data):
        print('[%s t=%s] %s' % (stream, t, data[:600]))
print('---- LAST 5 records ----')
for stream, t, data in msgs[-5:]:
    data = data.encode().decode('unicode_escape', errors='replace')
    print('[%s t=%s] %s' % (stream, t, data[:600]))
