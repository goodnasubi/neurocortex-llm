#!/bin/bash
# Rust実装とPyTorch版の出力一致を検証する（12.11.2節）。
# 参照は WSL2 の torch 2.14 で取る（Jetson の CPU ビルドは数値的に信頼できないため）。
set -e
cd "$(dirname "$0")/.."
BIN="${BIN:-$HOME/.cache/ncrs-target/x86_64-unknown-linux-musl/release/neurocortex-rs}"
for ref in "$@"; do
  echo "### 参照: $ref"
  "$BIN" verify --ref "$ref"
done
