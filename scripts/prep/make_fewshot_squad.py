import sys
N = eval(sys.argv[1]) if len(sys.argv) >= 2 else 10
N = int(N)

import json
data = json.load(open('/mnt/data/squad2/train-v2.0.json', 'r'))
data['data'] = data['data'][:1]
data['data'][-1]['paragraphs'] = data['data'][-1]['paragraphs'][:N]
num_examples = sum([len(x['qas']) for x in data['data'][-1]['paragraphs']])

data = json.dump(data, open(f'/mnt/data/squad2/trainfew{num_examples}-v2.0.json', 'w'))
print(f'Created dataset with {num_examples} question answer pairs')
