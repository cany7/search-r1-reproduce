#!/usr/bin/env bash
# 停掉 BM25 检索服务。
pkill -f "retrieval/retrieval_server.py" && echo "retrieval server stopped" || echo "retrieval server is not running"
