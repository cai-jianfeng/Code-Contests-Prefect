# %% setup
# from solutions_eval import dataset_read
# from datasets import Dataset, DatasetDict
# from tqdm import tqdm

# # %% load datasets
# codecontests_data_path = "/aiarena/gpfs/Code-Contests/test/test_dataset.parquet"
# codecontests_plus_data_path = "/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset"

# codecontests_dataset = Dataset.from_parquet(codecontests_data_path)
# codecontests_plus_dataset = dataset_read(codecontests_plus_data_path, transform="codecontents_plus", split="test")

# # print("Code Contests Dataset:", codecontests_dataset)
# # print("Code Contests Plus Dataset:", codecontests_plus_dataset)

# # 创建一个字典以便快速查找 codecontests_dataset 中的元素
# id_to_element = {item['name'].split('.')[0]: item for item in codecontests_dataset}

# # 创建一个新的数据集列表
# merged_dataset = []

# # 合并属性时处理相同和不同的 key
# for item in tqdm(codecontests_plus_dataset):
#     # 获取当前元素的 id
#     element_id = item['id']

#     # 检查 id 是否存在于 codecontests_dataset 中
#     if element_id in id_to_element:
#         # 获取对应的元素
#         original_item = id_to_element[element_id]

#         # 创建一个新的合并字典
#         merged_item = {}

#         # 处理不同的 key
#         unique_keys_item = set(item.keys()) - set(original_item.keys())
#         unique_keys_original = set(original_item.keys()) - set(item.keys())

#         for key in unique_keys_item:
#             merged_item[key] = item[key]

#         for key in unique_keys_original:
#             merged_item[key] = original_item[key]

#         # 处理相同的 key
#         common_keys = set(item.keys()) & set(original_item.keys())
#         for key in common_keys:
#             merged_item[f"plus_{key}"] = item[key]
#             merged_item[f"original_{key}"] = original_item[key]

#         # 添加到新数据集中
#         merged_dataset.append(merged_item)

# dataset = Dataset.from_list(merged_dataset)

# dataset = DatasetDict({"test": dataset})

# dataset['test'].to_json("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset.json", orient="records", lines=True)

# # %% test
# dataset = Dataset.from_json("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset.json")
# # print(dataset)
# # %%
# '''
# Dataset({
#     features: ['checker', 'generator_cmd', 'correct_submissions', 'true_negative_rate', 'generator', 'true_positive_rate', 'memory_limit', 'incorrect_submissions', 'title', 'validator', 'cf_points', 'untranslated_description', 'is_description_translated', 'name', 'output_file', 'generated_tests', 'cf_index', 'difficulty', 'private_tests', 'input_file', 'cf_contest_id', 'cf_rating', 'cf_tags', 'public_tests', 'original_labels', 'plus_canonical_solution', 'original_canonical_solution', 'original_content', 'original_description', 'original_time_limit', 'plus_test', 'original_test', 'original_incorrect_solutions', 'original_id', 'original_solutions', 'original_source'],
#     num_rows: 161
# })

# plus_incorrect_solutions == incorrect_submissions
# plus_solutions == correct_submissions
# plus_time_limit == original_time_limit * 1000
# memory_limit_bytes == memory_limit * 1000 * 1000
# plus_labels == original_labels
# plus_content == original_content
# plus_description == original_description
# plus_test == test_cases
# plus_id == original_id
# plus_source == original_source
# '''

# # 删除 plus_incorrect_solutions, plus_solutions, plus_time_limit, memory_limit_bytes, plus_labels, plus_content, plus_description, plus_test, plus_id, plus_source

# dataset = dataset.remove_columns([
#     'plus_incorrect_solutions',
#     'plus_solutions',
#     'plus_time_limit',
#     'memory_limit_bytes',
#     'plus_labels',
#     'plus_content',
#     'plus_description',
#     'plus_test',
#     'plus_id',
#     'plus_source'
# ])

# # %%
# import requests
# def sandbox_call(api_path, json_data):
#     """
#     调用远程的 sandbox API 来执行代码。
#     """
#     res = requests.post(
#         api_path,
#         json=json_data,
#     )

#     return res.json()

# '''
# plus_canonical_solution 和 original_canonical_solution 的结构为：
# {
#     'cpp': xxx,
#     'java': xxx,
#     'python': xxx,
# }
# '''

# # 将 original_labels, original_content, original_description, original_time_limit, original_incorrect_solutions, original_id, original_solutions, original_source 重命名为 labels, content, description, time_limit, incorrect_solutions, id, solutions, source
# # dataset = dataset.rename_columns({
# #     'original_labels': 'labels',
# #     'original_content': 'content',
# #     'original_description': 'description',
# #     'original_time_limit': 'time_limit',
# #     'original_incorrect_solutions': 'incorrect_solutions',
# #     'original_id': 'id',
# #     'original_solutions': 'solutions',
# #     'original_source': 'source'
# # })

# def save_canonical_solution(sample):
#     # 对于 plus_canonical_solution 和 original_canonical_solution, 每种语言保留一个

#     # 1. 获取 test_cases 里的第一个 input 即可
#     test_case_input = sample['test_cases'][0]['input']

#     # 2. 对于 plus_canonical_solution 和 original_canonical_solution 的每种语言, 调用 sandbox_call 并检查返回结果, 保留返回结果为 True 的语言 (优先保留 plus_canonical_solution 的语言)
#     canonical_solutions = {}
#     for lang in ['cpp', 'java', 'python']:
#         plus_solution = sample['plus_canonical_solution'].get(lang)
#         original_solution = sample['original_canonical_solution'].get(lang)

#         if plus_solution:
#             res = sandbox_call('http://10.244.213.170:8080/run_code', {
#                 'language': lang,
#                 'code': plus_solution,
#                 'stdin': test_case_input
#             })
#             if res.get('status') == 'Success':
#                 canonical_solutions[lang] = plus_solution
#                 continue  # 如果 plus_solution 成功, 则不再检查 original_solution
        
#         if original_solution:
#                 res = sandbox_call('http://10.244.213.170:8080/run_code', {
#                     'language': lang,
#                     'code': original_solution,
#                     'stdin': test_case_input
#                 })
#                 if res.get('status') == 'Success':
#                     canonical_solutions[lang] = original_solution
#                     continue
        
#         # 3. 如果 plus_canonical_solution 和 original_canonical_solution 都没有成功, 则输出警告
#         if lang not in canonical_solutions:
#             print(f"Warning: No valid solution found for language {lang} in sample {sample['name']}")
    
#     # # 4. 确保 canonical_solutions 至少包含一种语言的代码
#     # assert canonical_solutions, f"No valid canonical solution found for sample {sample['name']}"
#     # 5. 如果 canonical_solutions 为空, 则输出警告
#     if not canonical_solutions:
#         print(f"Warning: No valid canonical solution found for sample {sample['name']}")
    
#     return canonical_solutions

# dataset_filter = []
# for sample in tqdm(dataset):
#     canonical_solution = save_canonical_solution(sample)
#     # 过滤 dataset 中元素的 canonical_solution 为 {} 的元素
#     if canonical_solution:
#         sample['canonical_solution'] = canonical_solution
#         dataset_filter.append(sample)

# dataset = Dataset.from_list(dataset_filter)
# # %% 保存 dataset
# dataset.save_to_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered")
# dataset = DatasetDict({"test": dataset})
# dataset['test'].to_json("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered.json", orient="records", lines=True)

# # %% 重新加载保存的数据集 - json
# dataset = Dataset.from_json("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered.json")
# # print(dataset)
# # %% 重新加载保存的数据集 - disk
# dataset = Dataset.load_from_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered")
# # print(dataset)

# # %%
# from datasets import load_dataset

# dataset = load_dataset("/aiarena/group/llmgroup/Code-Contests-Plus", "4x")

# from openai import OpenAI
# # 设置 OpenAI API 密钥
# API_BASE = "https://lonlie.plus7.plus/v1"
# API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

# client = OpenAI(base_url=API_BASE, api_key=API_KEY)

# response = client.responses.create(
#     model="gpt-5",
#     reasoning={"effort": "low"},
#     instructions="Talk like a pirate.",
#     input="Are semicolons optional in JavaScript?",
# )

# print(response.output_text)

# %%
