#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python analysis/p0/p0_2_format_kv.py --model Qwen/Qwen3-4B      --fmt chat --tag inst_chat --retrieval
python analysis/p0/p0_2_format_kv.py --model Qwen/Qwen3-4B-Base --fmt raw  --tag base_raw  --retrieval
python analysis/p0/p0_2_format_kv.py --model Qwen/Qwen3-4B      --fmt raw  --tag inst_raw
python analysis/p0/p0_2_format_kv.py --model Qwen/Qwen3-4B-Base --fmt chat --tag base_chat
