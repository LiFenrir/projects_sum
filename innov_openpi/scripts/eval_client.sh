#!/bin/bash
# 训练中评估客户端:等 .eval_ready 信号 → 起 RoboDojo 客户端连训练进程内服务器
set -euo pipefail

ROBODOJO_ROOT="/home/kemove/INNOV/sim/RoboDojo"
READY_FILE="$1"     # <ckpt_root>/.eval_ready
DONE_FILE="$2"      # <ckpt_root>/.eval_done
EVAL_GPU="${3:-1}"
EVAL_TIMEOUT="${4:-1800}"
EVAL_NUM="${EVAL_NUM:-20}"
SEED="${EVAL_SEED:-0}"

# eval_policy.sh 用裸 python,需 RoboDojo conda 环境(isaaclab)在 PATH 最前
export PATH="/home/kemove/miniconda3/envs/RoboDojo/bin:${PATH}"
# Omniverse Kit 非交互运行需预设 EULA 接受(与 install.sh 一致)
export OMNI_KIT_ACCEPT_EULA=YES

wait_ready() {
    local deadline=$(( $(date +%s) + EVAL_TIMEOUT ))
    while [[ ! -f "${READY_FILE}" ]]; do
        if [[ $(date +%s) -ge ${deadline} ]]; then
            echo "[eval_client] timeout waiting for ${READY_FILE}" >&2
            exit 2
        fi
        sleep 5
    done
}

wait_ready
port="$(grep -oP '(?<=^port=)\d+' "${READY_FILE}")"
step="$(grep -oP '(?<=^step=)\d+' "${READY_FILE}")"
task="$(grep -oP '(?<=^task=)\S+' "${READY_FILE}")"
echo "[eval_client] ready: step=${step} task=${task} port=${port}"

result_dir="${ROBODOJO_ROOT}/eval_result/${task}/innov_pi05"
mkdir -p "${result_dir}"   # 首次评估目录不存在时 find 会非零退出触发 set -e
before_count="$(find "${result_dir}" -name _result.json 2>/dev/null | wc -l)"

status="failed"
success_rate=-1
eval_time=-1

if (cd "${ROBODOJO_ROOT}" && bash scripts/robodojo.sh client \
    --policy-dir XPolicyLab/policy/innov_pi05 \
    --task "${task}" \
    --env-cfg arx_x5_eval \
    --policy-host 127.0.0.1 \
    --policy-port "${port}" \
    --ckpt external \
    --eval-num "${EVAL_NUM}" \
    --seed "${SEED}" \
    --env-gpu "${EVAL_GPU}" \
    --action-type joint); then

    after_count="$(find "${result_dir}" -name _result.json 2>/dev/null | wc -l)"
    if [[ "${after_count}" -gt "${before_count}" ]]; then
        newest="$(find "${result_dir}" -name _result.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
        eval_time="$(python3 -c "import json;print(json.load(open('${newest}'))['eval_time'])")"
        total_success="$(python3 -c "import json;d=json.load(open('${newest}'));print(d.get('total_success', 0))")"
        total_num="$(python3 -c "import json;d=json.load(open('${newest}'));print(d.get('total_num', 1))")"
        success_rate="$(python3 -c "print(${total_success}/${total_num})")"
        status="ok"
    fi
fi

python3 -c "
import json, sys
json.dump({'status': '${status}', 'success_rate': ${success_rate}, 'eval_time': ${eval_time}},
          open('${DONE_FILE}', 'w'), ensure_ascii=False)
"
echo "[eval_client] done: status=${status} success_rate=${success_rate} eval_time=${eval_time}"
