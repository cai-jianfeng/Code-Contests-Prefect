# %%
from datasets import load_dataset
import requests
from tqdm import tqdm

config = {
  'locale': "en",
  'run_timeout': 20,
  'extra': {
    'is_freeform': False
  },
  'dataset_type': "HumanEvalDataset"
}

# Get dataset data in sandbox format
data = list(load_dataset("/aiarena/gpfs/openai_humaneval", split="test"))

# %%
results = []

for sample in tqdm(data):
    config['provided_data'] = sample
    completion = sample['canonical_solution']
    res = requests.post('http://10.244.20.188:8080/submit', json={
        'dataset': 'humaneval_python',
        'id': '',
        'completion': completion,
        'config': config
    })

    result = res.json()
    results.append({
        'id': sample['task_id'],
        'result': result
    })
# %%
accepted = 0
for res in tqdm(results):
    res = res['result']
    if res.get("accepted", False):
        accepted += 1

# 计算准确率
accuracy = accepted / len(results) if len(results) > 0 else 0

# 打印准确率结果
print(f"Submission accuracy: {accuracy:.2%}")
# %%
