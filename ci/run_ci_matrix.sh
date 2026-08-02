#!/usr/bin/env bash
#
# 多 docker 并发编译矩阵: 每个 combo (CANN×torch/torch_npu) 用预构建镜像
# fa-npu-ci:<name>, 并发跑 `python setup.py build`, 汇总 per-combo pass/fail。
# 目的: 检查代码在各 CANN/torch 版本组合下能否编译通过 (编译/语法/兼容问题)。
#
# 前置: 先用 ci/build_matrix_images.sh 构建各 combo 镜像。
# 编译不需要 NPU, 只需 CANN toolkit + bisheng (镜像内已具备)。
#
# 并发安全:
#   - 各 combo 容器挂载同一仓库源码 (构建只读源码), 用 --build-base=/tmp/build 把
#     构建产物隔离到容器内临时目录, 互不冲突。
#   - 子模块由本脚本预先初始化一次 (单容器), 并对每个编译容器设
#     FLASH_ATTN_SKIP_SUBMODULE_INIT=1 跳过 setup.py 内的 git submodule 调用,
#     避免 N 个容器并发写 .git/config 损坏。
#
# 环境变量:
#   MATRIX_FILE           (默认 ci/build_matrix.tsv)
#   IMAGE_PREFIX          (默认 fa-npu-ci)
#   CI_MATRIX_MAX_JOBS    (默认 0=不限) 并发容器数上限 (内存吃紧时调小, 如 3)
#   CI_DOCKER_PRIVILEGED  (默认 true)
#   FLASH_ATTN_BUILD_VERSION (默认 all) 编译哪些 API 代 (all/v2/v3/v4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATRIX_FILE="${MATRIX_FILE:-$REPO_ROOT/ci/build_matrix.tsv}"
IMAGE_PREFIX="${IMAGE_PREFIX:-fa-npu-ci}"
CI_MATRIX_MAX_JOBS="${CI_MATRIX_MAX_JOBS:-0}"
CI_DOCKER_PRIVILEGED="${CI_DOCKER_PRIVILEGED:-true}"
LOG_DIR="${REPO_ROOT}/build/matrix-logs"
# 架构过滤: 非空时只跑 tsv 第 7 列 (arch) 匹配的 combo。用于多机器分架构跑:
# 950 机器设 ARCH_FILTER=x86_64, 910B 机器设 ARCH_FILTER=aarch64, 各跑各的。
ARCH_FILTER="${ARCH_FILTER:-}"

log() { printf '[matrix-build] %s\n' "$*"; }
die() { printf '[matrix-build][ERROR] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
[ -f "$MATRIX_FILE" ] || die "matrix file not found: $MATRIX_FILE"

privileged_args=()
[ "$CI_DOCKER_PRIVILEGED" = "true" ] && privileged_args+=(--privileged)

read_combos() {
  if [ -n "$ARCH_FILTER" ]; then
    awk -F'|' -v arch="$ARCH_FILTER" '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ {next}
      NF >= 7 && $7 == arch {print}
      NF < 7 {print}
    ' "$MATRIX_FILE"
  else
    awk -F'|' '/^[[:space:]]*#/ || /^[[:space:]]*$/ {next} {print}' "$MATRIX_FILE"
  fi
}

COMBOS=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  COMBOS+=("$line")
done < <(read_combos)
[ "${#COMBOS[@]}" -gt 0 ] || die "no combo in $MATRIX_FILE"

# 从 combo 行解析镜像名: 若第 8 列 (image) 非空则直接用该镜像 (预构建), 否则用 $IMAGE_PREFIX:$name
combo_image() {
  local line="$1" name img
  name="${line%%|*}"
  img="$(printf '%s' "$line" | awk -F'|' '{print $8}')"
  if [ -n "$img" ]; then
    printf '%s\n' "$img"
  else
    printf '%s\n' "${IMAGE_PREFIX}:${name}"
  fi
}

mkdir -p "$LOG_DIR"

# 检查镜像是否都已构建
missing=()
for line in "${COMBOS[@]}"; do
  name="${line%%|*}"
  img="$(combo_image "$line")"
  docker image inspect "$img" >/dev/null 2>&1 || missing+=("$name ($img)")
done
if [ "${#missing[@]}" -gt 0 ]; then
  die "missing matrix images: ${missing[*]}; run 'bash ci/build_matrix_images.sh' first"
fi

log "combos: ${#COMBOS[@]}"
if [[ "$CI_MATRIX_MAX_JOBS" =~ ^[0-9]+$ ]] && [ "$CI_MATRIX_MAX_JOBS" -gt 0 ]; then
  log "max jobs: $CI_MATRIX_MAX_JOBS"
else
  log "max jobs: unlimited"
fi
log "logs dir: $LOG_DIR"

# ---------- 1. 预初始化子模块 (单容器一次) ----------
first_name="${COMBOS[0]%%|*}"
first_img="$(combo_image "${COMBOS[0]}")"
log "pre-init submodule csrc/catlass (once, via $first_img)"
docker run --rm \
  "${privileged_args[@]}" \
  --network host \
  -v "$REPO_ROOT:/workspace/flash-attention-npu" \
  -w /workspace/flash-attention-npu \
  "$first_img" \
  bash -lc 'git config --global --add safe.directory "*" && git submodule update --init --recursive csrc/catlass' \
  || die "submodule pre-init failed"

# ---------- 2. 每个 combo 一个容器, 并发编译 ----------
build_one() {
  local line="$1" name logf rc img
  name="${line%%|*}"
  img="$(combo_image "$line")"
  logf="$LOG_DIR/${name}.log"
  : > "$logf"
  echo "[matrix-build] >>> $name ($img) -> $logf"
  set +e
  docker run --rm \
    "${privileged_args[@]}" \
    --network host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    -e FLASH_ATTN_FORCE_BUILD=TRUE \
    -e FLASH_ATTN_SKIP_SUBMODULE_INIT=1 \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -w /workspace/flash-attention-npu \
    "$img" \
    bash -lc 'git config --global --add safe.directory "*" && python3 setup.py build --build-base=/tmp/build' \
    > "$logf" 2>&1
  rc=$?
  echo "$rc" > "$LOG_DIR/${name}.rc"
}

for line in "${COMBOS[@]}"; do
  if [[ "$CI_MATRIX_MAX_JOBS" =~ ^[0-9]+$ ]] && [ "$CI_MATRIX_MAX_JOBS" -gt 0 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$CI_MATRIX_MAX_JOBS" ]; do sleep 1; done
  fi
  build_one "$line" &
done
set +e; wait; set -e

# ---------- 3. 汇总结果 ----------
echo ""
log "=== results ==="
overall=0
for line in "${COMBOS[@]}"; do
  name="${line%%|*}"
  rc="$(cat "$LOG_DIR/${name}.rc" 2>/dev/null || echo 1)"
  if [ "$rc" -eq 0 ]; then
    printf '  %-28s PASS\n' "$name"
  else
    printf '  %-28s FAIL (rc=%s, see %s)\n' "$name" "$rc" "$LOG_DIR/${name}.log"
    echo "  ----- tail of ${name}.log (last 80 lines) -----"
    tail -n 80 "$LOG_DIR/${name}.log" 2>/dev/null | sed 's/^/  /' || true
    echo "  ----- end of ${name}.log -----"
    overall=1
  fi
done

exit $overall
