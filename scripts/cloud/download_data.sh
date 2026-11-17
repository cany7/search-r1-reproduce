set -e
source /workspace/.secrets/env.sh
export PATH=/home/ray/.local/bin:$PATH
D=/workspace/search-r1/project/data; mkdir -p $D; cd $D
date; 
uvx --from 'huggingface_hub[hf_transfer]' hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --local-dir nq_hotpotqa_train
uvx --from 'huggingface_hub[hf_transfer]' hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir bm25-index
uvx --from 'huggingface_hub[hf_transfer]' hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir corpus
date; pigz -dk corpus/wiki-18.jsonl.gz; date; du -sh *; echo DL_DONE
