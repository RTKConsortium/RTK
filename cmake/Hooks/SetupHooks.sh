#!/usr/bin/env bash
#==========================================================================
#
#   Copyright Insight Software Consortium
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#==========================================================================*/


# Run this script to set up the git hooks for committing changes to ITK.
# For more information, see:
#   https://www.itk.org/Wiki/ITK/Git#Hooks
#   https://www.itk.org/Wiki/Git/Hooks

egrep-q() {
  egrep "$@" >/dev/null 2>/dev/null
}

die() {
  echo 'failure during hook setup' 1>&2
  echo '-------------------------' 1>&2
  echo '' 1>&2
  echo "$@" 1>&2
  exit 1
}

if test -z "$GIT_DIR"; then
  export GIT_DIR=$(git rev-parse --git-dir)
fi

u=$(git rev-parse --git-dir)
cd "$u/hooks"

# We need to have a git repository to do a pull.
if ! test -d ./.git; then
  git init || die "Could not run git init."
fi

# Grab the hooks.
# Use the local hooks if possible.
echo "Pulling the hooks..."
remote_hooks="https://github.com/SimonRit/RTK.git hooks"
if GIT_DIR=.. git for-each-ref refs/remotes/origin/hooks 2>/dev/null | \
  egrep-q 'refs/remotes/origin/hooks$'; then
  git fetch .. remotes/origin/hooks ||  \
  echo "Couldn't use local hooks. Using remote..."
  git fetch $remote_hooks
else
  git fetch $remote_hooks
fi &&

git reset --hard FETCH_HEAD -- || die "Failed to install hooks"
cd ../..

# Disable the 'hooks' branch submodule check.
# We have a check that blocks addition of submodules.
#git config hooks.submodule false

# Set up KWStyle hook.
echo "Setting up the KWStyle hook..."
git config hooks.KWStyle.conf "cmake/KWStyle/RTK.kws.xml"
git config hooks.KWStyle.overwriteRulesConf "cmake/KWStyle/RTKOverwrite.txt"
git config hooks.KWStyle true

echo "Done."
