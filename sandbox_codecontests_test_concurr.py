# %% setup
from datasets import load_dataset
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import requests
# %% 加载 dataset 并初步 step up
config = {
  'language': "python",
  'locale': "en",
  'compile_timeout': 20,
  'run_timeout': 20,
  'dataset_type': "CommonOJDataset"
}

data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="train"))

# %% submit solutions to sandbox (并行提交)

URL = "http://10.244.53.180:8080/submit"
DATASET = "code_contests_train"
MAX_WORKERS = 16

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""

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
    sample['test'] = sample['test']
    cfg["provided_data"] = sample

    # 2️⃣ 选出可用的 canonical solution
    completion = None
    for lang, sol in sample["canonical_solution"].items():
        if sol:
            cfg["language"] = lang
            completion = TEMPLATE.format(
                language=lang,
                solution=sol
            )
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
# %%
