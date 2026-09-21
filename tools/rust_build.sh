#!/bin/bash
# WSL2 上で Rust 推論実装をビルドする補助スクリプト。
# WSL2 には C コンパイラが無く sudo も使えないため、musl ターゲット + rust-lld で
# self-contained リンクする（12.10節「Rustもここ」）。
set -e
cd "$(dirname "$0")/../rust"
TC="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$TC/lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/ncrs-target}"
"$HOME/.cargo/bin/cargo" build --release --target x86_64-unknown-linux-musl "$@"
echo "$CARGO_TARGET_DIR/x86_64-unknown-linux-musl/release/neurocortex-rs"
