import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_run_out\qwen3-5-0-8b-ple-target-side-reader-p100.log'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
late = [(s, t, x.encode().decode('unicode_escape', errors='replace')) for s, t, x in msgs if float(t) > 215]
for stream, t, data in late:
    if stream == 'stdout' or 'Error' in data or 'error' in data or 'assert' in data or 'rejected' in data or 'frozen load' in data or 'decoder blocks' in data or 'Exception' in data:
        print('[%s t=%s] %s' % (stream, t, data[:700]))
        print('---')
