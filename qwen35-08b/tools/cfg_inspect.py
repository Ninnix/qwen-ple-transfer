import json

D = r'C:\Users\NICOLO~1.DEV\AppData\Local\Temp\opencode\tok'
tc = json.load(open(D + '\\t35-08b-cfg.json', encoding='utf-8'))
print('top-level keys:', sorted(tc.keys()))
print('model_type:', tc.get('model_type'))
print('architectures:', tc.get('architectures'))
print('auto_map:', json.dumps(tc.get('auto_map'), indent=1)[:600])
print('tie_word_embeddings:', tc.get('tie_word_embeddings'))
tx = tc.get('text_config', {})
print('text keys:', sorted(tx.keys())[:20])
print('text model_type:', tx.get('model_type'), '| hidden:', tx.get('hidden_size'),
      '| layers:', tx.get('num_hidden_layers'), '| vocab:', tx.get('vocab_size'))
print('top hidden:', tc.get('hidden_size'), '| top layers:', tc.get('num_hidden_layers'))
