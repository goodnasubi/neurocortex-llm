#!/bin/bash
# Colab の Web UI と同じ keep-alive API を5分おきに叩く。kernel への exec だけでは「アクセス」と見なされない可能性がある
cd "$(dirname "$0")"
while true; do
  r=$(timeout 60 ~/.local/share/uv/tools/google-colab-cli/bin/python ka_api.py 2>&1 | tail -1)
  echo "$(date '+%F %T') $r" >> keepalive.log
  sleep 300
done
