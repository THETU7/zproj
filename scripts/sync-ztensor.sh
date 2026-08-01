#!/usr/bin/env bash
#
# Refresh the vendored ztensor (third_party/ztensor) from an upstream ztensor
# checkout. Mirrors llama.cpp's scripts/sync-ggml.sh.
#
# Usage:
#   $ cd /path/to/zproj
#   $ ./scripts/sync-ztensor.sh [path-to-ztensor-checkout]
#
# The default upstream is the sibling checkout ../ztensor. Only the
# build-essential file set is vendored (CMakeLists.txt, VERSION, cmake/,
# include/, src/) -- the same kind of file set llama.cpp vendors for ggml.
#
# NOTE: uncommitted changes under third_party/ztensor are overwritten by this
# script. Commit or push them to upstream first (see push-ztensor.sh).
set -euo pipefail

ZPROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${1:-${ZPROJ_ROOT}/../ztensor}"
VENDORED="${ZPROJ_ROOT}/third_party/ztensor"
LAST_FILE="${ZPROJ_ROOT}/scripts/sync-ztensor.last"

if [ ! -d "${UPSTREAM}/.git" ]; then
    echo "error: upstream ztensor checkout not found at '${UPSTREAM}'" >&2
    exit 1
fi
if [ ! -f "${VENDORED}/CMakeLists.txt" ]; then
    echo "error: vendored ztensor not found at '${VENDORED}'" >&2
    exit 1
fi

# Never clobber local work in the vendored tree.
if [ -n "$(git -C "${ZPROJ_ROOT}" status --porcelain -- third_party/ztensor)" ]; then
    echo "error: third_party/ztensor has uncommitted changes." >&2
    echo "       Commit them in zproj or push them to upstream first" >&2
    echo "       (./scripts/push-ztensor.sh)." >&2
    exit 1
fi

file_list="$(mktemp)"
trap 'rm -f "${file_list}"' EXIT
git -C "${UPSTREAM}" ls-files -- CMakeLists.txt VERSION cmake include src > "${file_list}"

echo "Syncing $(wc -l < "${file_list}") files from ${UPSTREAM}"
rsync -a --delete --files-from="${file_list}" "${UPSTREAM}/" "${VENDORED}/"

git -C "${UPSTREAM}" log -1 --format=%H > "${LAST_FILE}"
echo "Vendored ztensor is now ${UPSTREAM} @ $(cat "${LAST_FILE}")"
echo "Done"
