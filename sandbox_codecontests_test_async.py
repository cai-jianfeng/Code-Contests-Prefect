# # %%
# from datasets import load_dataset
# import requests
# from tqdm import tqdm

# config = {
#   'language': "cpp",
#   'locale': "en",
#   'compile_timeout': 20,
#   'run_timeout': 20,
#   'dataset_type': "CommonOJDataset"
# }

# # Get dataset data in sandbox format
# data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="test"))

# # %%
# config['provided_data'] = data
# prompts = requests.post('http://localhost:8080/get_prompts', json={
#   'dataset': 'code_contests_test',
#   'config': config
# }).json()

# # 调用 OpenAI API 获取每个 prompt 的 completion
# from openai import OpenAI
# api_key = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"
# base_url = "https://lonlie.plus7.plus/v1"
# # client = OpenAI(api_key=api_key, base_url=base_url)
# completions = []
# for prompt in tqdm(prompts[:1]):
#     # response = client.chat.completions.create(
#     #     model="gpt-4o",
#     #     messages=[
#     #         {"role": "system", "content": "You are a helpful assistant that provides code solutions."},
#     #         {"role": "user", "content": prompt['prompt']}
#     #     ]
#     # )
#     # completions.append(response.choices[0].message.content)
#     completion = """
# Sure, here's the complete C++ implementation of the solution:

# ```cpp
# #include <iostream>
# #include <vector>
# #include <string>
# #include <algorithm>
# using namespace std;

# // Custom comparator for sorting
# bool customComparator(const pair<string, int>& a, const pair<string, int>& b) {
#     const string& strA = a.first;
#     const string& strB = b.first;
#     size_t m = strA.size();
#     for (size_t i = 0; i < m; ++i) {
#         if (strA[i] != strB[i]) {
#             // Odd-indexed (1-based) -> compare lexicographically ascending
#             if ((i + 1) % 2 == 1) {
#                 return strA[i] < strB[i];
#             } 
#             // Even-indexed (1-based) -> compare lexicographically descending
#             else {
#                 return strA[i] > strB[i];
#             }
#         }
#     }
#     return false; // If strings are identical, their order is already defined
# }

# int main() {
#     int n, m;
#     cin >> n >> m;
#     vector<pair<string, int>> books(n);

#     // Read input
#     for (int i = 0; i < n; ++i) {
#         cin >> books[i].first; // The book title
#         books[i].second = i + 1; // Save the original index (1-based)
#     }

#     // Sort using the custom comparator
#     sort(books.begin(), books.end(), customComparator);

#     // Output the sorted indices
#     for (const auto& book : books) {
#         cout << book.second << " ";
#     }
#     cout << endl;

#     return 0;
# }
# ```

# ### Explanation of the Code

# 1. **Input Parsing**:
#    - First, the integers `n` (number of books) and `m` (length of book titles) are read.
#    - A vector of pairs `books` is used to store each book title along with its original index.

# 2. **Custom Comparator**:
#    - A custom comparator `customComparator` is defined to sort the book titles according to the asc-desc-ending order.
#    - The comparator iterates character by character and determines the order:
#      - For odd positions (1-based), characters are compared in ascending (`<`) order.
#      - For even positions (1-based), characters are compared in descending (`>`) order.
#    - The loop exits early as soon as a difference is found between the characters.

# 3. **Sorting**:
#    - The `std::sort` function is used to order the books using the custom comparator.

# 4. **Output**:
#    - After sorting, the original indices of the books (stored in the second element of the pair) are printed in the new order.

# ### Example

# #### Input:
# ```
# 5 2
# AA
# AB
# BB
# BA
# AZ
# ```

# #### Execution:
# - The custom comparator will sort the books based on their characters in asc-desc-ending order.
# - Resulting order of indices: `5 2 1 3 4`.

# #### Output:
# ```
# 5 2 1 3 4
# ```

# ### Complexity
# - Sorting complexity is \(O(n \log n \cdot m)\), where \(n\) is the number of books and \(m\) is the length of each book title.
# - This is sufficient for the given constraint \(n \cdot m \leq 10^6\).

# ### How to Compile and Run
# - Save the code in a file named `main.cpp`.
# - Compile the code using: `g++ -std=c++17 -O2 main.cpp -o main`.
# - Run the compiled program: `./main`.
# - Provide input through standard input or redirection (e.g., using `< input.txt`).
# """
#     completions.append(completion)
# # %%
# results = []

# for completion, sample in tqdm(zip(completions, data[:1])):
#     # config['provided_data'] = sample
#     # if sample['canonical_solution']['cpp']:
#     #     completion = sample['canonical_solution']['cpp']
#     #     res = requests.post('http://localhost:8080/submit', json={
#     #         'dataset': 'code_contests_test',
#     #         'id': '',
#     #         'completion': completion,
#     #         'config': config
#     #     })
#     #     result = res.json()
#     #     results.append({
#     #         'id': sample['id'],
#     #         'result': result
#     #     })
#     config['provided_data'] = sample
#     res = requests.post('http://localhost:8080/submit', json={
#         'dataset': 'code_contests_test',
#         'id': '',
#         'completion': completion,
#         'config': config
#     })

#     print(f'result: {res.json()}')
#     break
# # %%

# accepted = 0
# for res in tqdm(results):
#     res = res['result']
#     if res.get("accepted", False):
#         accepted += 1

# # 计算准确率
# accuracy = accepted / len(results) if len(results) > 0 else 0

# # 打印准确率结果
# print(f"Submission accuracy: {accuracy:.2%}")
# # %%

'''
# %% setup
from datasets import load_dataset

from tqdm import tqdm
import requests

# %% 加载 dataset 并初步 step up
config = {
  'language': "cpp",
  'locale': "en",
  'compile_timeout': 20,
  'run_timeout': 20,
  'dataset_type': "CommonOJDataset"
}

data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="train"))

# config['provided_data'] = data

# prompts = requests.post('http://localhost:8080/get_prompts', json={
#   'dataset': 'code_contests_train',
#   'config': config
# }).json()

# print(prompts[0])

# %% submit solutions to sandbox (串行提交)
results = []

template = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""

for sample in tqdm(data[:1]):
    # print(sample)
    config['provided_data'] = sample
    canonical_solutions = sample['canonical_solution']
    completion = None
    # for language, solution in canonical_solutions.items():
    #     if solution:
    #         config['language'] = language
    #         completion = template.format(language=language, solution=solution)
    #         break
    solution = """
#include <iostream>
#include <string>

int main() {
    int T;
    std::cin >> T;

    while (T--) {
        std::string S;
        std::cin >> S;

        int balance = 0;
        bool isBalanced = true;

        for (char ch : S) {
            if (ch == '(') {
                balance++;
            } else if (ch == ')') {
                balance--;
            }

            // If balance goes negative, it means there are more closing brackets than opening ones at that point.
            if (balance < 0) {
                isBalanced = false;
                break;
            }
        }

        // If the balance is not zero at the end, then we have unmatched brackets.
        if (balance != 0) {
            isBalanced = false;
        }

        if (isBalanced) {
            std::cout << "YES" << std::endl;
        } else {
            std::cout << "NO" << std::endl;
        }
    }

    return 0;
}
"""
#     solution = """
# #include <iostream>
# #include <stack>
# #include <string>
# using namespace std;

# int main() {
#     int T;
#     cin >> T;
    
#     while (T--) {
#         string s;
#         cin >> s;
#         stack<char> st;
#         bool balanced = true;
        
#         for (char ch : s) {
#             if (ch == '(') {
#                 st.push(ch);
#             } else if (ch == ')') {
#                 if (st.empty()) {
#                     balanced = false;
#                     break;
#                 }
#                 st.pop();
#             }
#             // Note: If other types of brackets are added, handle them here.
#         }
        
#         if (!st.empty()) {
#             balanced = false;
#         }

#         cout << (balanced ? "YES" : "NO") << endl;
#     }
    
#     return 0;
# }
# """
    language = "cpp"
    completion = template.format(language=language, solution=solution)
    config['language'] = language
    if completion is None:
        print(f"No solution found for sample {sample['id']}")
        continue
    res = requests.post('http://localhost:8080/submit', json={
        'dataset': 'code_contests_train',
        'id': '',
        'completion': completion,
        'config': config
    })
    result = res.json()
    results.append(result)
    # print("Result: ", result)
    # if result.get("accepted"):
    #     print("Sample: ", sample)
    #     print("Completion: ", completion)
    #     print("Result: ", result)
        # break
    for test in result.get("tests", []):
        print(test)
        # if not test.get("passed"):
        #     # print(test['exec_info']['run_result'])
        #     # print(test['test_info']['output'])
        #     print(test)
'''

# %% setup
from datasets import load_dataset
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio  # pip install tqdm>=4.66.0

# %% 加载 dataset 并初步 step up
config = {
  'language': "python",
  'locale': "en",
  'compile_timeout': 20,
  'run_timeout': 20,
  'dataset_type': "CommonOJDataset"
}

data = list(load_dataset("/aiarena/gpfs/FusedCodeContests", split="train"))

# %% submit solutions to sandbox (异步并发提交)
SANDBOX_URL = "http://10.244.53.180:8080/submit"
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
        sample['test'] = sample['test'][:1]  # 只测试第一个测试用例
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

if __name__ == "__main__":
    # 假设 data 与 config 已经在作用域里
    final_results = asyncio.run(run_all(data, config))