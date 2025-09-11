
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from tqdm import tqdm

# dataset = load_dataset("/aiarena/group/llmgroup/Code-Contests-Plus", "3x")

# dataset = dataset['train']

import json

data_path = "/aiarena/group/llmgroup/Code-Contests-Plus/ccplus_5x_resharded/test_cases_test.json"

print(f"Loading data from {data_path}...")
with open(data_path, 'r') as f:
    dataset = json.load(f)

codecontests_data_path = "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered_with_1_3_test_cases"

data = load_from_disk(codecontests_data_path)

'''
data_dict = [sample['id'] for sample in data]
# 创建一个字典来快速查找 dataset 中的样本
print("Creating ID mapping for dataset...")
dataset_id_map = {}
for i, sample in enumerate(tqdm(dataset)):
    if "Codeforces/" + sample['id'].replace("_", "/") in data_dict:
        dataset_id_map["Codeforces/" + sample['id'].replace("_", "/")] = sample

print(f"Dataset contains {len(dataset_id_map)} samples")
print(f"Data contains {len(data)} samples")

# 为 data 中的每个样本添加 test_cases 字段
updated_samples = []
missing_ids = []

print("Processing samples and copying test_cases...")
for i, sample in enumerate(tqdm(data)):
    sample_id = sample['id']
    
    if sample_id in dataset_id_map:
        sample_original = dataset_id_map[sample_id]
        # 复制 test_cases 字段
        sample_copy = dict(sample)
        sample_copy['3x_test_cases'] = sample_original['test_cases']
        updated_samples.append(sample_copy)
    else:
        missing_ids.append(sample_id)
        # 如果找不到对应的ID，仍然保留原样本但不添加test_cases
        updated_samples.append(dict(sample))

print(f"Successfully copied test_cases for {len(updated_samples) - len(missing_ids)} samples")
if missing_ids:
    print(f"Warning: {len(missing_ids)} samples not found in dataset:")
    print(f"Missing IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")

'''

# 为 data 中的每个样本添加 test_cases 字段
updated_samples = []
missing_ids = []

print("Processing samples and copying test_cases...")
for i, sample in enumerate(tqdm(data)):
    sample_id = sample['id']
    
    if sample_id in dataset:
        sample_original = dataset[sample_id]
        # 复制 test_cases 字段
        sample_copy = dict(sample)
        sample_copy['5x_test_cases'] = sample_original
        updated_samples.append(sample_copy)
    else:
        missing_ids.append(sample_id)
        # 如果找不到对应的ID，仍然保留原样本但不添加test_cases
        updated_samples.append(dict(sample))

print(f"Successfully copied test_cases for {len(updated_samples) - len(missing_ids)} samples")

if missing_ids:
    print(f"Warning: {len(missing_ids)} samples not found in dataset:")
    print(f"Missing IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")

# 创建新的数据集
updated_data = Dataset.from_list(updated_samples)

# 保存更新后的数据
output_path = "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered" + "_with_1_3_5_test_cases"
print(f"Saving updated data to {output_path}...")
updated_data.save_to_disk(output_path)

print("Done!")

# 重新加载保存的数据集
new_dataset = load_from_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered" + "_with_1_3_5_test_cases")

#     features: ['checker', 'generator_cmd', 'correct_submissions', 'true_negative_rate', 'test_cases', 'generator', 'true_positive_rate', 'memory_limit', 'incorrect_submissions', 'title', 'validator', 'cf_points', 'untranslated_description', 'is_description_translated', 'name', 'output_file', 'generated_tests', 'cf_index', 'difficulty', 'private_tests', 'input_file', 'cf_contest_id', 'cf_rating', 'cf_tags', 'public_tests', 'labels', 'plus_canonical_solution', 'original_canonical_solution', 'content', 'description', 'time_limit', 'original_test', 'incorrect_solutions', 'id', 'solutions', 'source', 'canonical_solution', '3x_test_cases'],
