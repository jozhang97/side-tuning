import json
import sys

subsample_rate = eval(sys.argv[1]) if len(sys.argv) > 1 else 2
data_path = "/mnt/data/squad2/train-v2.0.json"

data = json.load(open(data_path, 'r'))
data['data'] = data['data'][::subsample_rate]
with open(data_path.replace('train', f'train{subsample_rate}'), 'w') as f:
    json.dump(data, f)
