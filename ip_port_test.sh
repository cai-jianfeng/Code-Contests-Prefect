#!/bin/bash

ips=(
    "10.244.188.161"
    "10.244.40.170"
    "10.244.179.60"
    "10.244.204.74"
    "10.244.128.118"
    "10.244.81.204"
)

for ip in "${ips[@]}"; do
    for port in $(seq 8080 $((8080 + 127))); do
        # 执行 nc 并捕获结果
        result=$(nc -zv -w 1 "$ip" "$port" 2>&1)
        # 判断是否包含 success（不同系统输出可能是 "succeeded" 或 "open"）
        if [[ "$result" != *succeeded* && "$result" != *open* ]]; then
            echo "$result"
        fi
    done
done
