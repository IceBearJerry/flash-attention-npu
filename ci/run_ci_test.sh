#!/usr/bin/env bash
#
# 阶段2: NPU 自检 + 安装 + 测试 (容器内执行, 需要 NPU, 已加锁)
#   1. NPU 可用性自检
#   2. python setup.py install (复用阶段1 build/ 产物, 快速安装)
#   3. import 校验
#   4. 按 ci/example_st_cases.json 跑 Example ST (pytest)
#
# 由 ci/run_ci_container.sh 阶段2通过 docker run 调用 (绑卡 + 加锁)。
#
# 环境变量 (由 run_ci_container.sh 注入):
#   ASCEND_RT_VISIBLE_DEVICES   宿主机物理卡号
#   CI_MODE                     quick|full
#   CI_RUN_EXAMPLE_ST           true|false
#   CI_EXAMPLE_CASE_FILTER      只跑指定 case name (逗号分隔)
#   CI_CONTAINER_DEVICE         容器内逻辑设备号 (默认 0)

set -euo pipefail

REPO_ROOT="$(pwd)"
CASES_JSON="$REPO_ROOT/ci/example_st_cases.json"
DEVICE="${CI_CONTAINER_DEVICE:-0}"

# git safe.directory (容器内 root 操作宿主机 runner 用户的目录, 会触发 dubious ownership)
git config --global --add safe.directory "$REPO_ROOT"

log() { printf '[CI-test] %s\n' "$*"; }
die() { printf '[CI-test][ERROR] %s\n' "$*" >&2; exit 1; }

LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"
mkdir -p "$LOG_DIR"

log "repo=$REPO_ROOT device=$DEVICE mode=${CI_MODE:-quick}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
log "test phase start: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------- 1. NPU 自检 (需要卡) ----------
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"
python3 - <<'PY' || die "torch_npu not functional inside container (check --privileged / driver mount)"
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
assert torch_npu.npu.device_count() >= 1, "device_count==0; --privileged or driver mount missing?"
PY

# ---------- 2. 安装 (复用 build/ 产物, 不重新编译) ----------
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py install --skip-build (reuse build/ artifacts)"
python3 setup.py install --skip-build

log "import check"
python3 - <<'PY'
import flash_attn_npu_3
print("flash_attn_npu_3", flash_attn_npu_3.__version__)
PY

# ---------- 3. Example ST ----------
if [ "${CI_RUN_EXAMPLE_ST:-true}" != "true" ]; then
  log "CI_RUN_EXAMPLE_ST!=true, skip Example ST"
  exit 0
fi

command -v pytest >/dev/null 2>&1 || pip install pytest --quiet
python3 -c "import xdist" 2>/dev/null || pip install pytest-xdist --quiet
command -v jq >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq jq >/dev/null)

MODE="${CI_MODE:-quick}"
FILTER="${CI_EXAMPLE_CASE_FILTER:-}"
# case 内并行度: 用 pytest-xdist 多进程跑参数组合。默认 2, 用 CI_TEST_WORKERS 控制。
TEST_WORKERS="${CI_TEST_WORKERS:-2}"

run_case() {
  local name="$1" file="$2" kfilter="$3" args="$4"
  local logfile="$LOG_DIR/case_${name}.log"
  log ">>> case=$name file=$file k=${kfilter:-<none>} args=$args workers=$TEST_WORKERS (log=$logfile)"
  set +e
  # shellcheck disable=SC2086
  python3 -m pytest "$file" $args -n "$TEST_WORKERS" --dist=loadscope ${kfilter:+-k "$kfilter"} >"$logfile" 2>&1
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    log "<<< case=$name FAILED (pytest rc=$rc), tail of $logfile:"
    tail -n 30 "$logfile" 2>/dev/null | sed 's/^/    /'
    echo "$name" >> "$FAILED_FILE"
  else
    log "<<< case=$name OK"
  fi
}

select_cases() {
  # 有 FILTER 时选所有 case (不管 enabled), 否则只选 enabled=true
  if [ -n "$FILTER" ]; then
    jq -r '.cases[] | "\(.name)|\(.test_file)|\(.test_filter // "")|\(.pytest_args // "")"' "$CASES_JSON"
  else
    jq -r '.cases[] | select(.enabled==true) | "\(.name)|\(.test_file)|\(.test_filter // "")|\(.pytest_args // "")"' "$CASES_JSON"
  fi
}

filter_cases() {
  if [ -z "$FILTER" ]; then
    cat
  else
    local IFS_save="$IFS"
    IFS=','
    local -a want=($FILTER)
    IFS="$IFS_save"
    local line name w
    while IFS= read -r line; do
      name="${line%%|*}"
      for w in "${want[@]}"; do
        if [ "$name" = "$w" ]; then
          printf '%s\n' "$line"
          break
        fi
      done
    done
  fi
}

log "running Example ST (mode=$MODE filter=${FILTER:-<none>})"

# 直接模式: CI_TEST_DIRECT_FILE 指定时, 跳过 case 配置, 直接跑指定文件
if [ -n "${CI_TEST_DIRECT_FILE:-}" ]; then
  log "direct mode: file=$CI_TEST_DIRECT_FILE k=${CI_TEST_DIRECT_FILTER:-<none>} workers=$TEST_WORKERS"
  FAILED_FILE="$LOG_DIR/failed_cases.txt"
  : > "$FAILED_FILE"
  run_case "direct" "$CI_TEST_DIRECT_FILE" "${CI_TEST_DIRECT_FILTER:-}" "-vs"
  FAILED_CASES="$(cat "$FAILED_FILE" 2>/dev/null | tr '\n' ' ')"
  if [ -n "$FAILED_CASES" ]; then
    die "direct test FAILED"
  fi
  log "direct test passed"
  log "test phase end: $(date '+%Y-%m-%d %H:%M:%S')"
  exit 0
fi

selected="$(select_cases | filter_cases)"
if [ "$MODE" = "quick" ]; then
  selected="$(printf '%s\n' "$selected" | grep -v -iE '^[a-z0-9_]*bwd' || true)"
fi
log "selected cases:"
printf '%s\n' "$selected" | sed 's/^/  /'

if [ -z "$selected" ]; then
  die "no Example ST case selected (check ci/example_st_cases.json or CI_EXAMPLE_CASE_FILTER)"
fi

FAILED_FILE="$LOG_DIR/failed_cases.txt"
: > "$FAILED_FILE"
while IFS='|' read -r name file kfilter args; do
  [ -z "$name" ] && continue
  run_case "$name" "$file" "$kfilter" "$args" &
done <<< "$selected"

log "waiting for all cases to finish (parallel)..."
wait

FAILED_CASES="$(cat "$FAILED_FILE" 2>/dev/null | tr '\n' ' ')"
if [ -n "$FAILED_CASES" ]; then
  die "Example ST FAILED cases:$FAILED_CASES"
fi

log "all Example ST cases passed"
log "test phase end: $(date '+%Y-%m-%d %H:%M:%S')"
