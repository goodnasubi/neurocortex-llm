#!/bin/bash
# Rust実装の3モードを同条件でベンチする。トークン列も最適化フラグも完全に共通。
cd "$(dirname "$0")/.."
BIN="${BIN:-$HOME/.cache/ncrs-target/x86_64-unknown-linux-musl/release/neurocortex-rs}"
TOK="${TOK:-results/rust/reference_corpus.npz}"
ITERS="${ITERS:-30}"
"$BIN" calib
for m in event spiking-dense dense; do
  "$BIN" bench --mode "$m" --tokens "$TOK" --iters "$ITERS" "$@"
done
