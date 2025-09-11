#!/bin/bash
set -x
set -e

# apt update
# apt install net-tools
# apt install openssh-server -y
echo "$(hostname -I | awk '{print $1}')" >> /aiarena/gpfs/ip.txt
# mkdir ~/.ssh
# echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICVsuEuJSunj7zs1/dd4UVPLg3v8ipo4KOOo+9GXyeSn 1622613693@qq.com >> ~/.ssh/authorized_keys
# /etc/init.d/ssh start
# netstat -anolp
sed -i.bak '10s/\(max_runner_concurrency: \).*/\11/' /root/sandbox/sandbox/configs/local.yaml
sed -i.bak '120s/\(max_concurrency: \).*/\11/' /root/sandbox/sandbox/configs/local.yaml
cd /root/sandbox/scripts/
# 在make命令上一行插入端口解析逻辑
sed -i '/make run-online/i \
# 端口优先级：命令行第1参数 > _BYTEFAAS_RUNTIME_PORT > 8080\nPORT="${1:-${_BYTEFAAS_RUNTIME_PORT:-8080}}"\n' run.sh

# 再把原来的 PORT=... 部分替换成我们解析出的变量
sed -i 's/PORT=${_BYTEFAAS_RUNTIME_PORT:-8080}/PORT=${PORT}/' run.sh

cat > run_all.sh <<'EOF'
#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 端口列表：优先使用命令行入参，否则用内置默认
PORTS=("$@")
if [ ${#PORTS[@]} -eq 0 ]; then
  n=128
  start_port=8080
  PORTS=($(seq $start_port $((start_port + n - 1))))
fi

echo "Will start sandboxes on ports: ${PORTS[*]}"

IP=$(hostname -I | awk '{print $1}')

for p in "${PORTS[@]}"; do
  LOG_DIR="/aiarena/gpfs/sandbox_log/${IP}"
  mkdir -p "$LOG_DIR"

  echo "==> Starting sandbox on ${p} at ${IP}"
  "${DIR}/run.sh" "${p}" >"${LOG_DIR}/sandbox_${p}.out" 2>&1 &
done


# 等待所有后台任务结束（可根据需要去掉）
wait
EOF

chmod +x run_all.sh
bash run_all.sh