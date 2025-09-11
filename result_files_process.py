# %%
import os
import json
from tqdm import tqdm
data_ours = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new/"
file_paths = os.listdir(data_ours)
filter_data = []
error_data =[]
empty_data = []
for file_path in tqdm(file_paths):
    if file_path.endswith('.json'):
        file_path = os.path.join(data_ours, file_path)
        with open(file_path, 'r') as f:
            sample = json.load(f)
            # 将 error_data 中的样本对应的文件移动到 data_ours + "errors/" 目录下
            if sample['status'] == 'error':
                error_data.append(sample)
                error_file_path = os.path.join(data_ours, "error", os.path.basename(file_path))
                os.rename(file_path, error_file_path)
            else:
                # 如果 sample 的 corner_cases 字段为空，则将 sample['result'][-1]['corner_cases'] 赋值给它；并保存为原来的文件名
                # filter_data.append(sample)
                if not sample.get('corner_cases'):
                    # 如果 sample['result] 为空，则将 sample 对应的文件移动到 data_ours + "empty/" 目录下
                    if not sample['result']:
                        empty_data.append(sample)
                        empty_file_path = os.path.join(data_ours, "empty", os.path.basename(file_path))
                        os.rename(file_path, empty_file_path)
                    else:
                        for i in range(3):
                            corner_cases = sample['result'][-(i+1)]['corner_cases']
                            if corner_cases:
                                break
                        if not corner_cases:
                            empty_data.append(sample)
                            empty_file_path = os.path.join(data_ours, "empty", os.path.basename(file_path))
                            os.rename(file_path, empty_file_path)
                            continue
                        filter_data.append(sample)
                        sample['corner_cases'] = corner_cases
                        with open(file_path, 'w') as f:
                            json.dump(sample, f, indent=4, ensure_ascii=False)
                else:
                    filter_data.append(sample)
    else:
        print(f"File {file_path} is not a JSON file, skipping")
