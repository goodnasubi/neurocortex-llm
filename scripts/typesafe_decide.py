"""typesafe-ai（Jev System One）で判定する（AGENTS.md「意思決定」）。

認証キーはクラウド環境のネットワークシークレット（api.typesafe.ai 宛てにプロキシが付与）で渡すため、
このスクリプトはキーを扱わない。ローカルでは TYPESAFE_API_KEY があれば Authorization ヘッダーに付ける。

入力 JSON: {"state": "判断材料の文章", "questions": {名前: {"type": "choice", "criteria": {選択肢: 説明}}
                                               | {"type": "score", "criteria": [レベル0の説明, ...]}}}
使い方: python scripts/typesafe_decide.py input.json > output.json
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"


def decide(state: str, questions: dict, model: str = MODEL) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    print(json.dumps(decide(spec["state"], spec["questions"], spec.get("model", MODEL)), ensure_ascii=False, indent=2))
