# search-r1-reproduce

Search-R1 小规模复现。计划见 knowledge-base 的《Search-R1 复现计划》。

## 固定的版本

| 项 | 版本 |
|---|---|
| tinker-cookbook | `1991897`（2026-11-14），在 `pyproject.toml` 里固定 |
| Search-R1（只取 `retrieval_server.py`） | `598e61b` |
| SkyRL（云端才用） | `79d4c89`（2026-11-13） |
| Python | 3.12.14，`pyenv local` 指定，uv 只用它 |
| pyserini | 2.4.0，需要 JDK 21 |

## 目录

- `search_r1/`：从 cookbook 的 `recipes/search_tool` 复制来的 4 个文件，目前是未改动的原样。
- `retrieval/`：BM25 检索服务，独立的 uv 环境。`retrieval_server.py` 相对原文件只加了 `--host` / `--port`。
- `scripts/check_render.py`：环境自检，渲染两个模型的带工具 prompt。
- `data/`：数据、索引、语料，不进 git。

## 数据

```bash
cd data
hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --local-dir nq_hotpotqa_train
hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir bm25-index
hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir corpus
gunzip -k corpus/wiki-18.jsonl.gz
```

## 环境

```bash
uv sync                          # 主环境
(cd retrieval && uv sync)        # 检索环境
cp .env.example .env             # 然后填 OMLX_API_KEY
```

## 启动检索服务

```bash
./retrieval/start.sh             # 默认 127.0.0.1:8001
```

首次启动会把语料转成 Arrow 缓存（在 `~/.cache/huggingface/datasets`），需要几分钟。
