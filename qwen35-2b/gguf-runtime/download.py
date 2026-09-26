import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
OUT = TRACK / '.artifacts/gguf/base'
FROZEN = json.loads((TRACK / 'frozen.json').read_text())


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    repo, rev = FROZEN['target_model'], FROZEN['target_revision']
    info = HfApi(token=False).model_info(repo, revision=rev, files_metadata=True)
    assert info.sha == rev
    names = [s.rfilename for s in info.siblings if s.rfilename.endswith('.safetensors')]
    names += ['config.json', 'tokenizer.json', 'tokenizer_config.json',
              'model.safetensors.index.json', 'merges.txt', 'vocab.json',
              'chat_template.jinja', 'LICENSE']
    names = [n for n in names if n in {s.rfilename for s in info.siblings}]
    snapshot_download(repo, revision=rev, local_dir=OUT, allow_patterns=names,
                      token=False, max_workers=2)
    files = {}
    for s in info.siblings:
        if s.rfilename not in names:
            continue
        path = OUT / s.rfilename
        digest = sha(path)
        if s.lfs:
            assert digest == s.lfs.sha256, s.rfilename
        files[s.rfilename] = {'sha256': digest, 'bytes': path.stat().st_size}
    assert files['config.json']['sha256'] == FROZEN['target_config_sha256']
    assert files['tokenizer.json']['sha256'] == FROZEN['target_tokenizer_sha256']
    result = {'repo': repo, 'revision': rev, 'files': files}
    (HERE / 'source.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
