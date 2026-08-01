#!/usr/bin/env bash
#
# Push local modifications of the vendored ztensor (third_party/ztensor) back
# to an upstream ztensor checkout -- the reverse of sync-ztensor.sh.
#
# Usage:
#   $ cd /path/to/zproj
#   $ ./scripts/push-ztensor.sh [path-to-ztensor-checkout]
#
# The default upstream is the sibling checkout ../ztensor. Files are copied
# over and STAGED in the upstream checkout (git add), but NOT committed, so
# you can review the diff before committing and pushing to the ztensor repo.
set -euo pipefail

ZPROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${1:-${ZPROJ_ROOT}/../ztensor}"
VENDORED="${ZPROJ_ROOT}/third_party/ztensor"

if [ ! -d "${UPSTREAM}/.git" ]; then
    echo "error: upstream ztensor checkout not found at '${UPSTREAM}'" >&2
    exit 1
fi
if [ ! -f "${VENDORED}/CMakeLists.txt" ]; then
    echo "error: vendored ztensor not found at '${VENDORED}'" >&2
    exit 1
fi

file_list="$(mktemp)"
trap 'rm -f "${file_list}"' EXIT
git -C "${ZPROJ_ROOT}" ls-files -- third_party/ztensor \
    | sed 's#^third_party/ztensor/##' > "${file_list}"

echo "Copying $(wc -l < "${file_list}") files to ${UPSTREAM}"
rsync -a --files-from="${file_list}" "${VENDORED}/" "${UPSTREAM}/"

# Remove upstream-tracked files that no longer exist in the vendored copy.
git -C "${UPSTREAM}" ls-files -- CMakeLists.txt VERSION cmake include src \
    | while IFS= read -r f; do
        if [ ! -e "${VENDORED}/${f}" ]; then
            rm -f "${UPSTREAM}/${f}"
            echo "removed ${f}"
        fi
    done

git -C "${UPSTREAM}" add -A -- CMakeLists.txt VERSION cmake include src

echo
echo "Staged ztensor changes in ${UPSTREAM}:"
git -C "${UPSTREAM}" diff --cached --stat
echo
echo "Review:  git -C ${UPSTREAM} diff --cached"
echo "Commit:  git -C ${UPSTREAM} commit -m '...'"
echo "Push:    git -C ${UPSTREAM} push"
