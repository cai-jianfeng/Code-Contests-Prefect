"""
该代码的主要流程是:
1. 设计一个 Prompt 来引导 LLM 改进给定的每个 sample 的 generator
"""
from tqdm import tqdm
import base64
import requests
from corner_case_gen_parallel import DatasetProcessor, SandboxClient, OpenAIClient
import subprocess
from openai import OpenAI
import re
from datasets import Dataset

# 设置 OpenAI API 密钥
API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

SAMPLE_TEMPLATE = """
Inputs provided to you:
- description: The full problem statement (including input/output format, constraints, and special notes).
- generator: The current generator program (a complete C++ source file that includes and uses `testlib.h`; it should be directly compilable, e.g. `g++ -std=c++17 generator.cpp -o gen`, and when executed with command-line arguments such as `./gen -arg1 val1 ...` it produces input instances for the problem). The primary purpose of the generator is to generate problem inputs that follow the `description`.
- testlib.h: The content of `testlib.h` used by the generator.

Goal: Carefully read `description`, `generator`, and `testlib.h`. Identify shortcomings, missing edge-cases, reproducibility, performance, or compilation issues, and provide minimal, focused edits that improve the generator so its outputs better match the problem requirements and maximize input-space coverage.

Strict output format (must be followed exactly):
- Return zero or more patch blocks in the exact format below. Each patch block must contain only the code fragment to SEARCH for and the replacement fragment to REPLACE:

<<<<<<< SEARCH
<original code fragment to search for>
=======
<replacement fragment (the improved code)>
>>>>>>> REPLACE

- For each SEARCH block, you must strictly copy the exact content from the provided generator. Do NOT add or modify any characters, such as adding "-" or "+" at the beginning of lines. The SEARCH block must be an exact substring of the generator.
- For each REPLACE block, strictly follow the code format and ensure that after replacing the SEARCH content with the REPLACE content, the generator can be compiled and run directly.
- If multiple independent edits are needed, include multiple such blocks separated by a single blank line.
- Keep the patches as small as possible and include 1-3 lines of context before/after to ensure uniqueness.
- Pay close attention to code indentation, spaces, and line breaks; do not omit or alter them in the search/replace fragments.
- 
- Do not return a full-file rewrite unless absolutely necessary; if you do, explain why in EXPLANATION.

Special case (NO CHANGE):
- If the given `generator` already fully satisfies the `description` (correct, covers edge cases, reproducible, performant, and uses `testlib.h` appropriately), do NOT return patch blocks. Instead return the single line:

NO_CHANGE_NEEDED

followed by an EXPLANATION section (see below) that justifies why no change is required.

EXPLANATION (required after patches or NO_CHANGE_NEEDED):
- Begin this section with the line "EXPLANATION:" and provide a short justification (1-6 sentences) describing what was changed and why, or why no change is needed. If you added command-line options or modified behavior, list new arguments, defaults, and a short usage example.

Checklist for analysis and edits (use these to guide the patches):
1) Correctness: ensure outputs strictly follow the `description` input/output format and constraints (handle min/max, empty cases, special values).
2) Coverage: expand randomness/enumeration to cover boundaries, large values, repeated values, special-pattern cases and likely counterexamples.
3) Reproducibility & configurability: provide or preserve a seed option; allow controlling sample size and modes (e.g., "edge", "random", "stress").
4) Proper use of `testlib.h`: use testlib utilities instead of ad-hoc/uninitialized RNG or unsupported helpers.
5) Performance: avoid O(N^2) or huge memory usage when generating worst-case inputs for maximum allowed constraints.
6) Compileability: keep code compilable with a commonly used compiler (e.g., g++ -std=c++17) without undefined symbols or missing includes.
7) Minimal intrusion: prefer small focused edits (function-level or parameter-level) rather than full rewrites.

Output requirements before patches: list 3-6 concrete issues or risk points you found in the current generator, each as a short bullet with 1 line of explanation. Then emit the patch blocks (or NO_CHANGE_NEEDED) and the EXPLANATION.

Now the three variables you must refer to are:
--- Problem description:
{description}

--- testlib.h contents:
{testlib_data}

--- Current generator source:
{generator}

Strictly follow the output format above.
"""

import re

def apply_code_patches(original_code: str, model_response: str) -> str:
    """
    根据模型 response 中的多个 <<<<<<< SEARCH / ======= / >>>>>>> REPLACE 块
    自动对原始代码进行替换，返回改进后的代码。

    :param original_code: 原始代码字符串
    :param model_response: 模型返回的包含多个代码块的字符串
    :return: 替换后的代码
    """
    # 匹配每个替换块
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL
    )
    
    modified_code = original_code

    unmatched_block = []

    # 找到所有匹配的替换块
    for search_block, replace_block in pattern.findall(model_response):
        search_block = search_block.strip("\n")
        replace_block = replace_block.strip("\n")
        
        # 在代码中进行替换
        if search_block in modified_code:
            modified_code = modified_code.replace(search_block, replace_block)
        else:
            print(f"Warning: Search block not found in code:\n{search_block}\n")
            unmatched_block.append((search_block, replace_block))

    return modified_code, unmatched_block

dataset_processor = DatasetProcessor()
dataset = dataset_processor.read_dataset("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered", "test")

testlib_path = "testlib.h"

with open(testlib_path, 'r', encoding='utf-8') as file:
    testlib_data = file.read()

new_dataset = []
for idx, sample in enumerate(tqdm(dataset)):
    
    generator = sample['generator']
    description = sample['description']
    
    prompt = SAMPLE_TEMPLATE.format(
        description=description,
        testlib_data=testlib_data,
        generator=generator,
    )

    messages = [
            {"role": "system", "content": "You are a helpful assistant. You are an editor-style assistant specialized in automatically generating code-fix suggestions. You must strictly follow the user's instructions."},
            {"role": "user", "content": prompt}
        ]
    
    model_response = client.responses.create(
        model="o4-mini",
        input=messages,
        max_output_tokens=8192,
        reasoning={"effort": "low"},
    )

    generator_update, unmatched_blocks = apply_code_patches(generator, model_response.output_text)

    sample['generator_refined'] = generator_update
    sample['unmatched_blocks'] = unmatched_blocks
    sample['refine_message'] = model_response.output_text

    new_dataset.append(sample)

dataset = Dataset.from_list(new_dataset)
dataset.save_to_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered_refine_gen")

dataset_processor = DatasetProcessor()
dataset = dataset_processor.read_dataset("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered_refine_gen", "test")

testlib_path = "testlib.h"

with open(testlib_path, 'r', encoding='utf-8') as file:
    testlib_data = file.read()

new_dataset = []
num = 0
failed_num = 0
for idx, sample in enumerate(tqdm(dataset)):

    if not sample['unmatched_blocks']:
        new_dataset.append(sample)
        continue
    num += 1
    generator = sample['generator']
    description = sample['description']
    
    prompt = SAMPLE_TEMPLATE.format(
        description=description,
        testlib_data=testlib_data,
        generator=generator,
    )

    messages = [
            {"role": "system", "content": "You are a helpful assistant. You are an editor-style assistant specialized in automatically generating code-fix suggestions. You must strictly follow the user's instructions."},
            {"role": "user", "content": prompt}
        ]
    
    try:
        model_response = client.responses.create(
            model="o4-mini",
            input=messages,
            max_output_tokens=8192,
            reasoning={"effort": "low"},
        )
    except Exception as e:
        print(f"Error processing sample id: {sample['id']}, error: {e}")
        new_dataset.append(sample)
        continue

    generator_update, unmatched_blocks = apply_code_patches(generator, model_response.output_text)

    sample['generator_refined'] = generator_update
    sample['unmatched_blocks'] = unmatched_blocks
    sample['refine_message'] = model_response.output_text

    new_dataset.append(sample)

    if unmatched_blocks:
        failed_num += 1
        print(f"Failed sample id: {sample['id']}, total failed: {failed_num}/{num}")

print(f"Total samples processed: {num}, total failed: {failed_num}")

if failed_num:
    dataset = Dataset.from_list(new_dataset)
    dataset.save_to_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered_refine_gen")
