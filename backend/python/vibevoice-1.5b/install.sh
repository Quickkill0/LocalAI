#!/bin/bash
set -e

backend_dir=$(dirname $0)
if [ -d $backend_dir/common ]; then
    source $backend_dir/common/libbackend.sh
else
    source $backend_dir/../common/libbackend.sh
fi

# This is here because the Intel pip index is broken and returns 200 status codes for every package name, it just doesn't return any package links.
# This makes uv think that the package exists in the Intel pip index, and by default it stops looking at other pip indexes once it finds a match.
# We need uv to continue falling through to the pypi default index to find optimum[openvino] in the pypi index
# the --upgrade actually allows us to *downgrade* torch to the version provided in the Intel pip index
if [ "x${BUILD_PROFILE}" == "xintel" ]; then
    EXTRA_PIP_INSTALL_FLAGS+=" --upgrade --index-strategy=unsafe-first-match"
fi

# Use python 3.12 for l4t
if [ "x${BUILD_PROFILE}" == "xl4t12" ] || [ "x${BUILD_PROFILE}" == "xl4t13" ]; then
  PYTHON_VERSION="3.12"
  PYTHON_PATCH="12"
  PY_STANDALONE_TAG="20251120"
fi

installRequirements

# Install transformers with VibeVoice 1.5B support (PR #40546)
# This enables the long-form multi-speaker model
# NOTE: Do NOT install the standalone vibevoice package - it conflicts with this PR
echo "Installing transformers with VibeVoice 1.5B support from PR #40546..."
if [ "x${USE_PIP}" == "xtrue" ]; then
    pip install ${EXTRA_PIP_INSTALL_FLAGS:-} git+https://github.com/huggingface/transformers.git@refs/pull/40546/head
else
    uv pip install ${EXTRA_PIP_INSTALL_FLAGS:-} git+https://github.com/huggingface/transformers.git@refs/pull/40546/head
fi
