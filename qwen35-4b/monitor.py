import argparse
import fcntl
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / '.artifacts'
UNIT = 'qwengram-4b-monitor.timer'
PROMPT = '''Scheduled 30-minute Qwengram-4B check requested by the user.
Check live Kaggle status and the existing pipeline in /home/nicolo/qwen-ple-transfer.
Report the current stage, latest confirmed token count, throughput, last safely
persisted checkpoint, and any failure in this chat. Distinguish old progress from
current activity. Continue authorized infrastructure recovery if needed; preserve
the frozen experiment and do not start additional experiments. Do not create a
second monitoring schedule. Stop qwengram-4b-monitor.timer after the final report
is persisted, or if the user asks to stop monitoring. Never reveal credentials.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', required=True)
    args = parser.parse_args()
    lock = (ART / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = HERE / 'results/monitor.json'
    observed = HERE / 'results/monitor-status.json'
    delivered = json.loads(path.read_text()) if path.exists() else {}
    checked = json.loads(observed.read_text()) if observed.exists() else {}
    log = ART / 'pipeline.log'
    complete = log.exists() and '4B_FINAL_REPORT_PERSISTED; NO FURTHER EXPERIMENTS' in log.read_text()
    # One pending check is enough when the chat cannot run unattended.
    if delivered.get('queue_exit_code') == 0 and (
            not checked.get('checked_utc') or
            datetime.fromisoformat(delivered['checked_utc']) >
            datetime.fromisoformat(checked['checked_utc'])):
        if complete:
            subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True, timeout=15)
        print('MONITOR_CHECK_ALREADY_QUEUED', flush=True)
        return
    now = datetime.now(timezone.utc).isoformat()
    prompt = PROMPT
    if complete:
        prompt += '\nThe pipeline logged final report persistence. Verify it, summarize completion, and stop monitoring.\n'
    codex = shutil.which('codex')
    if not codex:
        raise SystemExit('Codex executable unavailable; monitor notification not queued')
    result = subprocess.run([codex, 'queue', '--thread', args.thread, '--message', prompt],
                            cwd=HERE.parent, capture_output=True, text=True, timeout=90)
    status = {'checked_utc': now, 'interval_seconds': 1800,
              'queue_exit_code': result.returncode, 'final_report_marker': complete,
              'delivery': 'existing Codex chat', 'timer': UNIT}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    os.replace(tmp, path)
    with (ART / 'monitor-history.jsonl').open('a') as history:
        history.write(json.dumps(status) + '\n')
    print(json.dumps(status), flush=True)
    if result.returncode:
        # CLI errors can contain local connection details; keep raw output private.
        private = ART / 'monitor-delivery-error.log'
        with os.fdopen(os.open(private, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
            f.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    if complete:
        subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True, timeout=15)


if __name__ == '__main__':
    main()
