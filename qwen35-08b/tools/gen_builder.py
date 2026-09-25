import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / 'qwen35-08b' / 'tools' / 'ple-dataset'
OUT = REPO / 'qwen35-08b' / 'ple-data'

manifest = json.loads((TMP / 'manifest.json').read_text())
shards = [l.strip() for l in (TMP / 'FILES.txt').read_text().splitlines() if l.strip()]
assert len(shards) == 33 and len(manifest['parts']) == 128
assert sorted(set(manifest['parts'].values())) == shards

MAN = json.dumps(manifest)
SH = json.dumps(shards)

def cell_md(src):
    return {"cell_type": "markdown", "id": "m%d" % abs(hash(src)) % 100000,
            "metadata": {}, "source": [l + "\n" for l in src.split("\n")]}

def cell_code(cid, lines):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}

setup = [
    "# Cell 0 - setup: environment credentials and kaggle CLI",
    "import os, subprocess, sys, json, hashlib, shutil, time",
    "from pathlib import Path",
    "HF = os.environ.get('HF_TOKEN')",
    "KG = os.environ.get('KAGGLE_API_TOKEN')",
    "if not HF or not KG:",
    "    try:",
    "        from kaggle_secrets import UserSecretsClient",
    "        user_secrets = UserSecretsClient()",
    "        HF = HF or user_secrets.get_secret('HF_TOKEN')",
    "        KG = KG or user_secrets.get_secret('KG_TOKEN')",
    "    except Exception:",
    "        pass",
    "assert HF and KG, 'no Kaggle/HF credentials'",
    "os.environ['KAGGLE_API_TOKEN'] = KG",
    "os.environ['HF_TOKEN'] = HF",
    "%pip install -q kaggle==2.2.4",
    "WORK=Path('/kaggle/working/ple-build'); STAGE=WORK/'stage'; META=WORK/'meta'",
    "STAGE.mkdir(parents=True, exist_ok=True); META.mkdir(parents=True, exist_ok=True)",
    "print('kaggle', __import__('kaggle').__version__, '| staging under', WORK)",
]
payload = [
    "# Cell 1 - payload: exact shard list + manifest (authored offline from pinned index)",
    "PLE_SRC='Qwen/Qwen3.8-Flash-Next-FP8'",
    "PLE_REV='236dfdf285828023ca3bcd3f37366c58a3469b13'",
    "DS_SLUG='ninnix/qwen38-flashnext-ple-fp8'",
    "SHARDS=" + SH,
    "MANIFEST=" + MAN,
    "assert len(SHARDS)==33 and len(MANIFEST['parts'])==128",
    "assert sorted(set(MANIFEST['parts'].values()))==SHARDS",
    "(META/'manifest.json').write_text(json.dumps(MANIFEST, indent=2)+'\\n')",
    "(META/'dataset-metadata.json').write_text(json.dumps({'id': DS_SLUG, 'title': 'Qwen3.8-Flash-Next FP8 PLE shards (pinned)', 'licenses': [{'name': 'unknown'}]}, indent=2)+'\\n')",
    "print('payload ready:', len(SHARDS), 'shards + manifest.json')",
]
create = [
    "# Cell 2 - plan: 11 independent datasets (replace-semantics-proof, no staging limits)",
    "print('\\n'.join('ninnix/qwen38-ple-p%02d <- %s' % (i, ', '.join(SHARDS[3*i:3*i+3])) for i in range(11)) + '\\nmanifest.json travels in every dataset')",
]
loop = [
    "# Cell 3 - per batch of 3: skip if dataset exists, else download -> size+sha verify -> CREATE one dataset.",
    "# Independent single-version datasets: no staging limits, no re-uploads, no processing waits.",
    "from huggingface_hub import HfApi, hf_hub_download",
    "api=HfApi(token=HF)",
    "_info=api.model_info(PLE_SRC, revision=PLE_REV, files_metadata=True)",
    "_meta={s.rfilename: (s.size, (getattr(s.lfs, 'oid', None) or getattr(s.lfs, 'sha256', None) or getattr(s.lfs, 'sha', None) if s.lfs else None)) for s in _info.siblings if s.rfilename in SHARDS}",
    "assert len(_meta)==33 and all(v[0] for v in _meta.values()), 'incomplete file metadata'",
    "made=[]",
    "for bi in range(0, len(SHARDS), 3):",
    "    tag='p%02d' % (bi//3); batch=SHARDS[bi:bi+3]; slug='ninnix/qwen38-ple-'+tag",
    "    _r=subprocess.run(['kaggle','datasets','files',slug],capture_output=True,text=True,timeout=300)",
    "    _listed={l.split()[0].split('/')[-1] for l in _r.stdout.splitlines() if '.safetensors' in l or 'manifest.json' in l}",
    "    if set(batch)|{'manifest.json'}<=_listed:",
    "        print('exists, skip:', slug, flush=True); made.append(slug); continue",
    "    B=STAGE/('batch-'+tag); B.mkdir(parents=True, exist_ok=True)",
    "    shutil.copy(META/'manifest.json', B/'manifest.json')",
    "    (B/'dataset-metadata.json').write_text(json.dumps({'id': slug, 'title': 'Qwen3.8-Flash-Next PLE '+tag+' (pinned)', 'licenses': [{'name': 'unknown'}]}, indent=2)+'\\n')",
    "    got=0",
    "    for f in batch:",
    "        p=hf_hub_download(PLE_SRC, f, revision=PLE_REV, token=HF, local_dir=str(B))",
    "        assert Path(p).name==f, Path(p)",
    "        sz=Path(p).stat().st_size",
    "        assert sz==_meta[f][0], (f, sz, _meta[f][0])",
    "        _h=hashlib.sha256(); _fh=open(p,'rb')",
    "        [_h.update(c) for c in iter(lambda: _fh.read(64*1024*1024), b'')]; _fh.close()",
    "        assert _meta[f][1] is None or _h.hexdigest()==_meta[f][1], (f, 'sha256 mismatch')",
    "        got+=sz",
    "    print('batch %s downloaded %.2f GiB' % (tag, got/1024**3), flush=True)",
    "    _vr=subprocess.run(['kaggle','datasets','create','-p',str(B)],capture_output=True,text=True,timeout=1800)",
    "    print((_vr.stdout or '')[-800:], flush=True)",
    "    assert _vr.returncode==0, ((_vr.stdout or '')+(_vr.stderr or ''))[-2000:]",
    "    shutil.rmtree(B)",
    "    made.append(slug)",
    "    print('dataset %s submitted (%d/11)' % (slug, len(made)), flush=True)",
    "print('ALL 11 DATASETS SUBMITTED')",
]
final = [
    "# Cell 4 - finalize: poll each dataset until its exact 4-file set is ready (20 min cap each)",
    "import time",
    "allfiles=set()",
    "for i in range(11):",
    "    slug='ninnix/qwen38-ple-p%02d' % i; want=set(SHARDS[3*i:3*i+3])|{'manifest.json'}",
    "    _t0=time.time()",
    "    while True:",
    "        _r=subprocess.run(['kaggle','datasets','files',slug],capture_output=True,text=True,timeout=300)",
    "        _listed={l.split()[0].split('/')[-1] for l in _r.stdout.splitlines() if '.safetensors' in l or 'manifest.json' in l}",
    "        if _listed==want: break",
    "        assert time.time()-_t0 < 1200, (slug, 'not ready', sorted(_listed))",
    "        time.sleep(60)",
    "    allfiles|=want",
    "    print(slug, 'ready', flush=True)",
    "assert allfiles==set(SHARDS)|{'manifest.json'}",
    "print('PLE DATASETS COMPLETE: 11 x (3 shards + manifest). Attach all via dataset_sources, then run the 10K real-PLE smoke')",
]

cells = [
    {"cell_type": "markdown", "id": "m-title", "metadata": {}, "source": [
        "# PLE dataset builder (Kaggle-side, CPU, nothing stages locally)\n",
        "Pushes 33 pinned FP8 PLE shards (~48.7 GiB) + manifest.json in 11 additive versions (3 shards each).\n",
        "Each shard is size-verified against the pinned HF revision before its version push.\n"]},
    cell_code("b-setup", setup),
    cell_code("b-payload", payload),
    cell_code("b-create", create),
    cell_code("b-loop", loop),
    cell_code("b-final", final),
]
nb = {"cells": cells,
      "metadata": {"kaggle": {"dataSources": [], "dockerImageVersionId": 31041,
                              "isGpuEnabled": False, "isInternetEnabled": True,
                              "language": "python", "sourceType": "notebook"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb['cells']:
    if c['cell_type'] == 'code':
        body = '\n'.join(l for l in ''.join(c['source']).splitlines() if not l.strip().startswith(('%pip', '!')))
        compile(body, c['id'], 'exec')
print('builder cells compile')

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'build_ple_dataset.ipynb').write_text(json.dumps(nb))
meta = {"code_file": "build_ple_dataset.ipynb", "competition_sources": [], "dataset_sources": [],
        "enable_gpu": "false", "enable_internet": "true",
        "id": "ninnix/qwen3-8-flash-next-ple-dataset-build", "is_private": "true", "kernel_sources": [],
        "kernel_type": "notebook", "language": "python",
        "title": "Qwen3.8 Flash-Next PLE dataset build"}
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('WROTE', OUT)
