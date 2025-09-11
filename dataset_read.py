# %% setup
from datasets import load_dataset
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import collections
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio  # pip install tqdm>=4.66.0

# %% codecontests dataset read and statistics

codecontests_dataset = load_dataset("/aiarena/gpfs/code_contests")

"""
['name',
 'description',
 'public_tests',
 'private_tests',
 'generated_tests',
 'source',
 'difficulty',
 'solutions',
 'incorrect_solutions',
 'cf_contest_id',
 'cf_index',
 'cf_points',
 'cf_rating',
 'cf_tags',
 'is_description_translated',
 'untranslated_description',
 'time_limit',
 'memory_limit_bytes',
 'input_file',
 'output_file']
"""

train_dataset = codecontests_dataset["train"]

solutions = []
incorrect_solutions = []

for train_data in train_dataset:
    solutions.append(len(train_data['solutions']['solution']))
    incorrect_solutions.append(len(train_data['incorrect_solutions']['solution']))

# === 2. 统计出现次数 ===
cnt_solutions = collections.Counter(solutions)
cnt_incorrect_solutions = collections.Counter(incorrect_solutions)

# 所有可能的取值（并排序，保持 x 轴有序）
all_keys = sorted(set(cnt_solutions) | set(cnt_incorrect_solutions))

# 把 Counter 结果映射到同一顺序的向量
freq_solutions = [cnt_solutions.get(k, 0) for k in all_keys]
freq_incorrections = [cnt_incorrect_solutions.get(k, 0) for k in all_keys]

freq_solutions_interval = [0] * 11

freq_incorrections_interval = [0] * 11

interval = 100

for key, freq in zip(all_keys, freq_solutions):
    if key <= interval * 1:
        freq_solutions_interval[0] += freq
    elif key <= interval * 2:
        freq_solutions_interval[1] += freq
    elif key <= interval * 3:
        freq_solutions_interval[2] += freq
    elif key <= interval * 4:
        freq_solutions_interval[3] += freq
    elif key <= interval * 5:
        freq_solutions_interval[4] += freq
    elif key <= interval * 6:
        freq_solutions_interval[5] += freq
    elif key <= interval * 7:
        freq_solutions_interval[6] += freq
    elif key <= interval * 8:
        freq_solutions_interval[7] += freq
    elif key <= interval * 9:
        freq_solutions_interval[8] += freq
    elif key <= interval * 10:
        freq_solutions_interval[9] += freq
    else:
        freq_solutions_interval[10] += freq

for key, freq in zip(all_keys, freq_incorrections):
    if key <= interval * 1:
        freq_incorrections_interval[0] += freq
    elif key <= interval * 2:
        freq_incorrections_interval[1] += freq
    elif key <= interval * 3:
        freq_incorrections_interval[2] += freq
    elif key <= interval * 4:
        freq_incorrections_interval[3] += freq
    elif key <= interval * 5:
        freq_incorrections_interval[4] += freq
    elif key <= interval * 6:
        freq_incorrections_interval[5] += freq
    elif key <= interval * 7:
        freq_incorrections_interval[6] += freq
    elif key <= interval * 8:
        freq_incorrections_interval[7] += freq
    elif key <= interval * 9:
        freq_incorrections_interval[8] += freq
    elif key <= interval * 10:
        freq_incorrections_interval[9] += freq
    else:
        freq_incorrections_interval[10] += freq

# === 3. 绘制并列柱状图 ===
# x = np.arange(len(all_keys))          # 0, 1, 2, ...
x = np.array([i for i in range(11)])
width = 0.35                          # 每组柱子的宽度

plt.figure(figsize=(10, 4))
plt.bar(x - width/2, freq_solutions_interval, width, label='solutions')
plt.bar(x + width/2, freq_incorrections_interval, width, label='incorrect_solutions')

plt.xlabel('count of solutions/incorrect_solutions')
plt.ylabel('Frequency')
plt.title('solutions and incorrect_solutions counts')
plt.xticks(x, [f"${i}*{interval}$" for i in range(1, 11)] + [f"$>10*{interval}$"])               # 用元素值作为刻度标签
plt.legend()
plt.tight_layout()
plt.show()

# %% codecontests+ dataset read and statstics

codecontests_plus_dataset = load_dataset("/aiarena/gpfs/Code-Contests-Plus", "default")

train_dataset = codecontests_plus_dataset["train"]


solutions = []
incorrect_solutions = []

for train_data in train_dataset:
    solutions.append(len(train_data['correct_submissions']))
    incorrect_solutions.append(len(train_data['incorrect_submissions']))

# === 2. 统计出现次数 ===
cnt_solutions = collections.Counter(solutions)
cnt_incorrect_solutions = collections.Counter(incorrect_solutions)

# 所有可能的取值（并排序，保持 x 轴有序）
all_keys = sorted(set(cnt_solutions) | set(cnt_incorrect_solutions))

freq_solutions_interval = [0] * 11

freq_incorrections_interval = [0] * 11

interval = 100

for key, freq in zip(all_keys, freq_solutions):
    if key <= interval * 1:
        freq_solutions_interval[0] += freq
    elif key <= interval * 2:
        freq_solutions_interval[1] += freq
    elif key <= interval * 3:
        freq_solutions_interval[2] += freq
    elif key <= interval * 4:
        freq_solutions_interval[3] += freq
    elif key <= interval * 5:
        freq_solutions_interval[4] += freq
    elif key <= interval * 6:
        freq_solutions_interval[5] += freq
    elif key <= interval * 7:
        freq_solutions_interval[6] += freq
    elif key <= interval * 8:
        freq_solutions_interval[7] += freq
    elif key <= interval * 9:
        freq_solutions_interval[8] += freq
    elif key <= interval * 10:
        freq_solutions_interval[9] += freq
    else:
        freq_solutions_interval[10] += freq

for key, freq in zip(all_keys, freq_incorrections):
    if key <= interval * 1:
        freq_incorrections_interval[0] += freq
    elif key <= interval * 2:
        freq_incorrections_interval[1] += freq
    elif key <= interval * 3:
        freq_incorrections_interval[2] += freq
    elif key <= interval * 4:
        freq_incorrections_interval[3] += freq
    elif key <= interval * 5:
        freq_incorrections_interval[4] += freq
    elif key <= interval * 6:
        freq_incorrections_interval[5] += freq
    elif key <= interval * 7:
        freq_incorrections_interval[6] += freq
    elif key <= interval * 8:
        freq_incorrections_interval[7] += freq
    elif key <= interval * 9:
        freq_incorrections_interval[8] += freq
    elif key <= interval * 10:
        freq_incorrections_interval[9] += freq
    else:
        freq_incorrections_interval[10] += freq

# === 3. 绘制并列柱状图 ===
# x = np.arange(len(all_keys))          # 0, 1, 2, ...
x = np.array([i for i in range(11)])
width = 0.35                          # 每组柱子的宽度

plt.figure(figsize=(7, 4))
plt.bar(x - width/2, freq_solutions_interval, width, label='solutions')
plt.bar(x + width/2, freq_incorrections_interval, width, label='incorrect_solutions')

plt.xlabel('Element value')
plt.ylabel('Frequency')
plt.title('Element counts in solutions and incorrect_solutions')
# plt.xticks(x, [f"{i}" for i in range(11)] + ["$10^2$", "$10^3$", "$>10^3$"])               # 用元素值作为刻度标签
plt.legend()
plt.tight_layout()
plt.show()

# %% 加载 dataset 并初步 step up
config = {
  'language': "python",
  'locale': "en",
  'compile_timeout': 20,
  'run_timeout': 20,
  'dataset_type': "CommonOJDataset"
}

data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="test"))

# config['provided_data'] = data

# prompts = requests.post('http://192.168.4.13:42313/get_prompts', json={
#   'dataset': 'code_contests_train',
#   'config': config
# }).json()

# %% submit solutions to sandbox (串行提交)
results = []

template = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""

for sample in tqdm(data[:10]):
    config['provided_data'] = sample
    canonical_solutions = sample['canonical_solution']
    completion = None
    result_dict = {}
    for language, solution in canonical_solutions.items():
        if solution:
            config['language'] = language
            completion = template.format(language=language, solution=solution)
            
            res = requests.post('http://192.168.4.13:42313/submit', json={
                'dataset': 'code_contests_train',
                'id': '',
                'completion': completion,
                'config': config
            })
            result = res.json()
            result_dict[language] = result
            if result.get("accepted", False):
                print(f"Sample {sample.get('id', 'unknown')} accepted in {language}.")
        else:
            result_dict[language] = {"error": "No solution provided for this language."}
    results.append(result_dict)

# %% 统计给定的 solutions 的正确性

accepted = 0
accepted_test_cases = 0
rejected_test_cases = 0
total_test_cases = 0
for res in tqdm(results):
    if res.get("accepted", False):
        accepted += 1
    # test = res.get("tests", [])[0]
    # if test:
    #     output = test.get("test_info", {}).get('output', {}).get('stdout', '')
    # else:
    #     print(res)
    # outputs = output.split("\n")
    # # 统计 outputs 中 YES 和 NO 的个数, 并确保 outputs 中只有 "YES" 和 "NO". 如果包含其他内容, 则输出该内容
    # if not all(o in ["YES", "NO"] for o in outputs):
    #     print(f"Unexpected output in sample {res.get('id', 'unknown')}: {output}")
    # accepted_test_cases += outputs.count("YES")
    # rejected_test_cases += outputs.count("NO")
    # total_test_cases += len(outputs)

# 计算准确率
# test_case_accuracy = accepted_test_cases / total_test_cases if total_test_cases > 0 else 0
# 统计提交样本的准确率
accuracy = accepted / len(results) if len(results) > 0 else 0

# 打印准确率结果
# print(f"Test case accuracy: {test_case_accuracy:.2%}")
print(f"Submission accuracy: {accuracy:.2%}")
# %% submit solutions to sandbox (并行提交)

URL = "http://192.168.4.13:42313/submit"
DATASET = "code_contests_train"
MAX_WORKERS = 16

base_config = {
    "language": "cpp",
    "locale": "en",
    "compile_timeout": 20,
    "run_timeout": 20,
    "dataset_type": "CommonOJDataset",
}

def submit_one(sample, idx):
    """
    将单个样本提交到评测服务，返回 dict:
    {
        'idx': idx,          # 原始顺序索引
        'success': bool,
        'data': json or err
    }
    """
    # 1️⃣ 深拷贝防止线程间写冲突
    cfg = copy.deepcopy(base_config)
    cfg["provided_data"] = sample

    # 2️⃣ 选出可用的 canonical solution
    completion = None
    for lang, sol in sample["canonical_solution"].items():
        if sol:
            cfg["language"] = lang
            completion = sol
            break
    if completion is None:
        return {"idx": idx, "success": False,
                "data": f"No solution for sample {sample['id']}"}

    payload = {
        "dataset": DATASET,
        "id": sample.get("id", ""),
        "completion": completion,
        "config": cfg,
    }

    try:
        # 可按需要调高 timeout，避免服务器慢时大量失败
        resp = requests.post(URL, json=payload, timeout=cfg["run_timeout"] + 10)
        resp.raise_for_status()
        return {"idx": idx, "success": True, "data": resp.json()}
    except Exception as e:
        return {"idx": idx, "success": False, "data": str(e)}

def parallel_submit(data):
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, sample in enumerate(data):
            futures.append(pool.submit(submit_one, sample, i))

        # tqdm 跟踪完成情况
        results_raw = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Submitting"):
            results_raw.append(fut.result())

    # 按原顺序整理
    results_raw.sort(key=lambda x: x["idx"])
    return [r["data"] for r in results_raw]

# 使用
results = parallel_submit(data)

# 如果只想要成功结果，可再过滤
successes = [r for r in results if isinstance(r, dict)]
errors    = [r for r in results if not isinstance(r, dict)]
# %% submit solutions to sandbox (codecontests)

data = list(load_dataset("/aiarena/gpfs/code_contests", split="train"))

FusedCodeContests_data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="train"))

sample = data[7]
fusedcodecontests_sample = FusedCodeContests_data[7]

config = {
  'language': "python",
  'locale': "en",
  'compile_timeout': 20,
  'run_timeout': 20,
  'dataset_type': "CommonOJDataset"
}

config['provided_data'] = fusedcodecontests_sample

completion = sample['solutions']['solution'][153]

res = requests.post('http://192.168.4.13:42313/submit', json={
    'dataset': 'code_contests_train',
    'id': '',
    'completion': completion,
    'config': config
})

# %% submit solutions to sandbox (异步并发提交)

SANDBOX_URL = "http://192.168.4.13:42313/submit"
CONCURRENCY = 16         # 并发协程上限

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""

async def _submit_one(session, sample, base_config, sem):
    """
    向沙盒提交一个 sample（可能含多语言实现），返回 result_dict。
    """
    async with sem:                       # 控制并发
        config = base_config.copy()       # 避免跨任务写同一个 dict
        config["provided_data"] = sample

        result_dict = {}
        canonical = sample["canonical_solution"]

        for language, solution in canonical.items():
            if not solution:              # 没有代码直接记错误
                result_dict[language] = {"error": "No solution provided."}
                continue

            config["language"] = language
            completion = TEMPLATE.format(language=language, solution=solution)

            payload = {
                "dataset": "code_contests_train",
                "id": "", # sample.get("id", ""),
                "completion": completion,
                "config": config,
            }

            try:
                async with session.post(SANDBOX_URL, json=payload) as resp:
                    result = await resp.json()
            except Exception as e:
                result = {"error": str(e)}

            result_dict[language] = result
            if result.get("accepted", False):
                print(f"Sample {sample.get('id','?')} accepted in {language}.")

        return result_dict


async def run_all(data, base_config):
    """
    高层调度：批量提交所有 sample 并收集结果。
    """
    # 全局并发控制器 & HTTP 连接池
    sem       = asyncio.Semaphore(CONCURRENCY)
    timeout   = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as sess:
        tasks = [asyncio.create_task(_submit_one(sess, s, base_config, sem)) for s in data]

        results = []
        # tqdm_asyncio.as_completed 自动在任务完成时更新进度条
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            results.append(await fut)

    return results

final_results = asyncio.run(run_all(data, config))
# %%
test_dataset = codecontests_dataset['test']

train_plus_dataset = codecontests_plus_dataset['train']

ids = {data['name'].split('.')[0]:data for data in test_dataset}

test_plus_dataset = []
for data in train_plus_dataset:
    if data['id'] in ids:
        data['solutions'] = ids[data['id']]['solutions']
        data['incorrect_solutions'] = ids[data['id']]['incorrect_solutions']
        test_plus_dataset.append(data)

# 将 test_plus_dataset 重新保存为常见的 dataset 的格式 (如 parquet)，使得可以直接使用 load_dataset 加载

from datasets import Dataset, DatasetDict, load_from_disk

test_plus_dataset = Dataset.from_list(test_plus_dataset)

test_plus_dataset = DatasetDict({"test": test_plus_dataset})

test_plus_dataset['test'].to_parquet("/aiarena/gpfs/Code-Contests-Plus/test/test_plus_dataset.parquet")


# test_plus_dataset.save_to_disk("/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset")

# # 加载新保存的数据集

# test_plus_dataset = load_from_disk("/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset")
# %% plot - train

# 给定 num cpus 和 consume times，绘制折线图

import matplotlib.pyplot as plt

num_cpus = [32, 64, 128, 256, 512]

consume_times = [8364.97, 4603.12, 3029.14, 1558.64, 883.39]
ideal_consume_times = [consume_times[0] / 2**i for i in range(len(num_cpus))]

plt.plot(num_cpus[:len(consume_times)], consume_times, 'bo-')
# 红色圆点标记数据点，并使用虚线表示
plt.plot(num_cpus, ideal_consume_times, 'ro', linestyle='dashed')
plt.plot(128, 2778.58, 'go')  # 标记 128 个 CPU 的实际消耗时间
# 在 128 处绘制一条垂直于 x 轴的虚线，并标注 multi-node
plt.axvline(x=128, color='gray', linestyle='--')
plt.text(128+65, 5000, 'multi-node begin', color='gray', ha='center', va='bottom')

plt.xlabel("num cpus")
plt.ylabel("consume time (seconds)")

plt.title("CodeContests Train Dataset (16 samples, 1 concur 1 runconcur)")

plt.show()

# %% plot - test

# 给定 num cpus 和 consume times，绘制折线图

import matplotlib.pyplot as plt

num_cpus = [190, 512]

consume_times = [76237.35, 35436.75]
ideal_consume_times = [consume_times[0] / (num_cpus[i] / num_cpus[0]) for i in range(len(num_cpus))]

plt.plot(num_cpus[:len(consume_times)], consume_times, 'bo-')
# 浅蓝色圆点标记数据点，并使用虚线表示
plt.plot(num_cpus, ideal_consume_times, 'o', color='skyblue', linestyle='dashed')

# 标记 512 个 CPU 的 (有保存结果) 实际消耗时间 (设置颜色为红色)

consume_times_save = [68008.61, 18565.00]
ideal_consume_times_save = [consume_times_save[0] / (num_cpus[i] / num_cpus[0]) for i in range(len(num_cpus))]
plt.plot(num_cpus, ideal_consume_times_save, 'o', color='salmon', linestyle='dashed')
plt.plot(num_cpus, consume_times_save, 'ro-')
# 标记 512 个 CPU 的 (没有保存结果) 实际消耗时间
plt.plot(512, 18824.53, 'go')

plt.xlabel("num cpus")
plt.ylabel("consume time (seconds)")

plt.title("CodeContests Test Dataset (165 samples, 1 concur 1 runconcur)")

plt.show()

# %% plot - test-plus

# 给定 num cpus 和 consume times，绘制折线图

import matplotlib.pyplot as plt

num_cpus = [190, 512]

# 标记 190 个 CPU 的 (有保存结果，plus) 实际消耗时间
# 绘制颜色为黄色，橙色，棕色的点
plt.plot(190, 13768.33, 'o', color='yellow')
plt.plot(190, 42894.39, 'o', color='orange')
plt.plot(190, 12382.29, 'o', color='brown')

plt.xlabel("num cpus")
plt.ylabel("consume time (seconds)")

plt.title("CodeContests Test-Plus Dataset (161 samples, 1 concur 1 runconcur)")

plt.show()
# %%
