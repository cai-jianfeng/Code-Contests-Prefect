"""
该脚本用于从指定的结果文件夹中加载测试结果，并将其与数据集进行关联。它会筛选出所有通过测试的解决方案，并将这些解决方案和语言信息存储到数据集中。最终，处理后的数据集将被保存为 Parquet 格式。
"""
# %% setup
import os
import json
from tqdm import tqdm
from solutions_eval import dataset_read
import requests
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import trange

LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]


def load_results(result_folder):
    result_files = os.listdir(result_folder)
    results = {}

    for result_file in tqdm(result_files):
        if result_file.endswith(".json") and "checker" not in result_file:
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    results[data.get("id", "")] = data

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    print(f"Loaded {len(results)} results from {result_folder}")
    return results

# %% main
if __name__ == "__main__":
    # result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_plus_test_results_190sandbox_190_1concur_1runconcur_190_collocate_queue_dir_161"
    result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_165"

    results = load_results(result_folder)

    # codecontents_data_path = "/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset"
    codecontents_data_path = "/aiarena/gpfs/code_contests"

    # dataset = dataset_read(codecontents_data_path, transform="codecontents_plus", split="test")
    dataset = dataset_read(codecontents_data_path, transform="codecontents", split="test")

    # 将 results 和 dataset 通过各自的 id 进行关联
    # 将 results 根据 id 进行排序
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    # 将 dataset 根据 id 进行排序
    sorted_dataset = sorted(dataset, key=lambda x: x['id'])

    saved_dataset = []
    for i in trange(len(sorted_results), desc="Processing results"):
        result_id, result_data = sorted_results[i]
        data = sorted_dataset[i]
        solution_results = result_data.get("solution_result", [])
        solutions = []
        languages = []
        assert len(solution_results) == len(data['solutions']['solution']), f"Mismatch in number of solutions for id {result_id}"
        for solution in solution_results:
            flag = True
            single_result = solution['result']
            tests = single_result.get("tests", [])
            for idx, test in enumerate(tests):
                if test['exec_info'].get('status') == "Failed":
                    flag = False
                    break
            if flag:
                solutions.append(solution['solution'])
                languages.append(LANGUAGE.index(solution['language']))
                
        data['solutions'] = {
            'language': languages,
            'solution': solutions
        }

        incorrect_solution_results = result_data.get("incorrect_solution_result", [])
        incorrect_solutions = []
        incorrect_languages = []
        assert len(incorrect_solution_results) == len(data['incorrect_solutions']['solution']), f"Mismatch in number of incorrect solutions for id {result_id}"
        for solution in incorrect_solution_results:
            flag = True
            single_result = solution['result']
            tests = single_result.get("tests", [])
            for idx, test in enumerate(tests):
                if test['exec_info'].get('status') == "Failed":
                    flag = False
                    break
            if flag:
                incorrect_solutions.append(solution['solution'])
                incorrect_languages.append(LANGUAGE.index(solution['language']))
        data['incorrect_solutions'] = {
            'language': incorrect_languages,
            'solution': incorrect_solutions
        }

    dataset = Dataset.from_list(sorted_dataset)

    dataset = DatasetDict({"test": dataset})

    dataset['test'].to_parquet("/aiarena/gpfs/Code-Contests/test/test_dataset.parquet")

    # 重新读取 parquet 文件并验证
    loaded_dataset = Dataset.from_parquet("/aiarena/gpfs/Code-Contests/test/test_dataset.parquet")
    assert len(loaded_dataset) == len(dataset['test']), "Loaded dataset size does not match original"
    print("Parquet file saved and loaded successfully. Number of samples:", len(loaded_dataset))
