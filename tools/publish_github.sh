#!/usr/bin/env bash
# Push swadesh to GitHub. Prerequisite (one click): create an EMPTY public repo
# https://github.com/new  ->  owner apeters247, name "swadesh", no README/license.
# This machine's SSH key is already authorized for that account.
set -euo pipefail
cd "$(dirname "$0")/.."
git remote get-url origin >/dev/null 2>&1 || git remote add origin git@github.com:apeters247/swadesh.git
git push -u origin master
echo "Pushed. Next: enable GitHub Pages (Settings > Pages > branch master, /docs) for docs/playground/,"
echo "then add PyPI trusted publishing and run: git tag v0.1.0 && git push --tags"
