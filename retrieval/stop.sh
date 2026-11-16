#!/usr/bin/env bash
# Stop the BM25 retrieval server.
pkill -f "retrieval/retrieval_server.py" && echo "retrieval server stopped" || echo "retrieval server is not running"
