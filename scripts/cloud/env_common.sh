# Source the Pod's credentials without printing them. Run helpers from the synced checkout.
CLOUD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export POD_ROOT=${POD_ROOT:-/workspace}
source "${POD_ENV_FILE:-$POD_ROOT/.secrets/env.sh}" || return 1
PROJ=$(cd "$CLOUD_DIR/../.." && pwd) || return 1
export PROJ
export PATH="/home/ray/.local/bin:$PATH"
export UV_PYTHON=${UV_PYTHON:-/home/ray/anaconda3/bin/python3.12}
export RETRIEVAL_JAVA_HOME=${RETRIEVAL_JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}
export RETRIEVAL_URL=${RETRIEVAL_URL:-http://127.0.0.1:8001}
export TINKER_BASE_URL=${TINKER_BASE_URL:-http://127.0.0.1:8000}
export TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy}
export SKYRL=${SKYRL:-$POD_ROOT/search-r1/SkyRL}
export CKPT_BASE=${CKPT_BASE:-$POD_ROOT/ckpts/skyrl}
export RAY_TMPDIR=${RAY_TMPDIR:-$POD_ROOT/ray_tmp}
export SEARCH_R1_DATA_DIR=${SEARCH_R1_DATA_DIR:-$PROJ/data}
mkdir -p "$POD_ROOT/logs" "$RAY_TMPDIR"
