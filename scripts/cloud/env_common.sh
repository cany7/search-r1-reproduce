# Common environment for all services on the pod. Source this; never echo secrets.
source /workspace/.secrets/env.sh
export PATH=/home/ray/.local/bin:$PATH
export UV_PYTHON=/home/ray/anaconda3/bin/python3.12
export RETRIEVAL_JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export RETRIEVAL_URL=http://127.0.0.1:8001
export TINKER_API_KEY=tml-dummy
export TINKER_BASE_URL=http://127.0.0.1:8000
export PROJ=/workspace/search-r1/project
export SKYRL=/workspace/search-r1/SkyRL
mkdir -p /workspace/logs
# Ray builds a per-session copy of the SkyRL venv in its temp dir; keep it off the 30 GB system disk.
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p $RAY_TMPDIR
