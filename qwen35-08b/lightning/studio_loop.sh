#!/bin/sh
# Studio entry: persistent resume loop for free-T4 restarts. Exits 0 only when train reports done.
set -u
: "${HF_TOKEN:?set HF_TOKEN in Studio secrets}"
ROOT="${LIGHTNING_PERSISTENT_DIR:-/teamspace/studios/this_studio/persistent}"
python sync_frozen.py || exit 1
python run.py --stage inspect --persistent "$ROOT" || exit 1
while true; do
  python run.py --stage train --persistent "$ROOT" && break
  echo "leg stopped (wall-guard or restart); relaunch resumes"
  sleep 5
done
python run.py --stage eval --persistent "$ROOT"
