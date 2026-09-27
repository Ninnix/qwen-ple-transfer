import json

from smoke import ART, HERE, run


def main():
    reference = json.loads((HERE / 'smoke.json').read_text())['Q6_K-cpu']['stdout']
    model = ART / 'release/QwenGram-4B-Q6_K.gguf'
    result = run('Q6_K-vulkan-ngl33', model, 'vulkan', 33)
    assert result['returncode'] == 0
    result.pop('stderr')
    result['exact_greedy_match'] = result['stdout'] == reference
    result['purpose'] = 'Check the earlier release workaround: keep the first decoder layer on CPU'
    (HERE / 'q6-fallback.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
