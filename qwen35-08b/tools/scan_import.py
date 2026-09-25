import re

P = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\kaggle_run_out\qwen3-5-0-8b-ple-target-side-reader-p100.log'
d = open(P, encoding='utf-8', errors='replace').read()
msgs = re.findall(r'"stream_name":"(stdout|stderr)","time":([\d.]+),"data":"(.*?)(?<!\\)"', d)
for stream, t, data in msgs:
    data = data.encode().decode('unicode_escape', errors='replace')
    if 'direct qwen3_5' in data or 'modeling import failed' in data or 'ModuleNotFound' in data or 'No module named' in data or 'ImportError' in data or 'in <module>' in data or '.py in ' in data and 'line' in data:
        print('[%s t=%s] %s' % (stream, t, data[:500]))
