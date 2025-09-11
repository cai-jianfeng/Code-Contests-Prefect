# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# test_s.py

# 读取给定日志文件（每行格式示例："[2025-09-11 18:30:55] xxx"），
# 找出相邻两行时间差超过设定阈值 T（秒）的情况，输出两行内容与行号。

# 用法示例：
#   python3 test_s.py /path/to/logfile --threshold 60
#   python3 test_s.py /path/to/logfile -t 2m

# 阈值支持带单位：s（秒）、m（分钟）、h（小时）。
# """

# import re
# import argparse
# from datetime import datetime
# from typing import Optional, Tuple

# TIMESTAMP_RE = re.compile(r'^\s*\[([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})\]\s*(.*)$')


# def parse_threshold(s: str) -> float:
#     """解析阈值字符串并返回秒数。支持纯数字或带单位：s/m/h。"""
#     s = s.strip().lower()
#     if not s:
#         raise ValueError("空阈值")
#     if s.endswith('s'):
#         return float(s[:-1])
#     if s.endswith('m'):
#         return float(s[:-1]) * 60.0
#     if s.endswith('h'):
#         return float(s[:-1]) * 3600.0
#     return float(s)


# def parse_line(line: str) -> Tuple[Optional[datetime], str]:
#     """解析一行，返回 (datetime 或 None, 原始行文本去除末尾换行)。"""
#     m = TIMESTAMP_RE.match(line)
#     text = line.rstrip('\n')
#     if not m:
#         return None, text
#     ts_text, rest = m.group(1), m.group(2)
#     try:
#         ts = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
#     except ValueError:
#         return None, text
#     return ts, text


# def find_gaps(path: str, threshold_s: float):
#     prev_ts: Optional[datetime] = None
#     prev_line: Optional[str] = None
#     prev_lineno: Optional[int] = None
#     skipped = 0

#     try:
#         f = open(path, 'r', encoding='utf-8')
#     except Exception as e:
#         print(f"无法打开文件 '{path}': {e}")
#         return

#     with f:
#         for idx, raw in enumerate(f, start=1):
#             ts, full_line = parse_line(raw)
#             if ts is None:
#                 skipped += 1
#                 # 不更新 prev_ts/prev_line，让下一个可解析时间戳继续与上一个可解析时间比较
#                 continue
#             if prev_ts is not None and prev_line is not None and prev_lineno is not None:
#                 delta = (ts - prev_ts).total_seconds()
#                 if delta > threshold_s:
#                     print(f"Gap (lines {prev_lineno}-{idx}) {delta:.3f}s")
#                     print(f"{prev_lineno}: {prev_line}")
#                     print(f"{idx}: {full_line}")
#                     print('')
#             prev_ts = ts
#             prev_line = full_line
#             prev_lineno = idx

#     if skipped:
#         print(f"[Info] 跳过 {skipped} 行无法解析时间戳的行。")


# def main():
#     parser = argparse.ArgumentParser(description="在日志中查找相邻行时间差超过阈值的片段。")
#     parser.add_argument('logfile', help='日志文件路径')
#     parser.add_argument('-t', '--threshold', default='60', help='阈值，默认 60（秒）。可带单位：s、m、h，例如 90, 2m, 1h')
#     args = parser.parse_args()

#     try:
#         threshold_s = parse_threshold(args.threshold)
#     except Exception as e:
#         print('无法解析阈值:', e)
#         return

#     find_gaps(args.logfile, threshold_s)


# if __name__ == '__main__':
#     main()

import os
from tqdm import tqdm
import shutil

folder = "./"

files = [f for f in os.listdir(folder) if os.path.isfile(f) and '.json' not in f]

for file in tqdm(files):
    file_path = os.path.join(folder, file)
    new_file_path = os.path.join('./ut_gen', file)
    shutil.copy(file_path, new_file_path)