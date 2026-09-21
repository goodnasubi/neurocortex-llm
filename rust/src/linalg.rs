//! 線形代数と活性化の基本演算。
//!
//! **公平性の方針**: 密な行列積とイベント駆動のgatherは、どちらも同じ水準で書く。
//! 具体的には (1) 内側ループは連続メモリを舐める形に揃える、(2) 4本のアキュムレータで
//! 依存連鎖を切って自動ベクトル化を効かせる、(3) 境界検査を外すために
//! スライスを固定長に切り出してから回す、を両方に適用する。片方だけ手を抜かない。

/// トークンあたりのMAC数カウンタ（12.6.3節の理論値との突き合わせ用）。
#[derive(Default, Clone, Copy, Debug)]
pub struct MacCounters {
    /// ブロック内の重み行列積：実際に実行したMAC数
    pub block_exec: u64,
    /// ブロック内の重み行列積：密に計算した場合のMAC数（理論上の分母）
    pub block_dense: u64,
    /// 出力ヘッド（常に密）
    pub head: u64,
    /// アテンションの状態演算（重みを持たない。常に密）
    pub attn_state: u64,
    /// 発火したユニット数の累計（発火率の実測に使う）
    pub spikes: u64,
    /// スパイクを取りうるユニット数の累計
    pub spike_slots: u64,
}

impl MacCounters {
    pub fn add(&mut self, o: &MacCounters) {
        self.block_exec += o.block_exec;
        self.block_dense += o.block_dense;
        self.head += o.head;
        self.attn_state += o.attn_state;
        self.spikes += o.spikes;
        self.spike_slots += o.spike_slots;
    }
}

/// SIMDの幅（AVX2 / NEON のどちらでも1ベクトルに収まる要素数）。
const W: usize = 8;
/// 独立したアキュムレータの本数（FMAのレイテンシを隠すためのILP）。
const U: usize = 4;

/// 長さ n の内積。`W` 要素幅 × `U` 本のアキュムレータに分解して、
/// 自動ベクトル化（AVX2/FMA, NEON）と命令レベル並列の両方を取る。
///
/// Rust は浮動小数点の再結合を許さないため、アキュムレータを明示的に分けないと
/// SIMD 化もパイプライン化もされない。これは密な側の性能に直結するので手を抜かない。
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [[0.0f32; W]; U];
    let step = W * U;
    let mut i = 0;
    while i + step <= n {
        for (u, accu) in acc.iter_mut().enumerate() {
            let o = i + u * W;
            for ((s, av), bv) in accu.iter_mut().zip(&a[o..o + W]).zip(&b[o..o + W]) {
                *s += av * bv;
            }
        }
        i += step;
    }
    while i + W <= n {
        for ((s, av), bv) in acc[0].iter_mut().zip(&a[i..i + W]).zip(&b[i..i + W]) {
            *s += av * bv;
        }
        i += W;
    }
    for u in 1..U {
        for w in 0..W {
            acc[0][w] += acc[u][w];
        }
    }
    let mut s = 0.0f32;
    for w in 0..W {
        s += acc[0][w];
    }
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// 入力行のブロック幅（レジスタブロッキング）。
const MR: usize = 4;
/// 出力方向のレジスタブロック幅（MR*NR 個のアキュムレータをレジスタに載せる）。
const NR: usize = 16;

/// 密な線形層: `out[m, n] = x[m, k] · W^T + bias[n]`
///
/// 重みは**転置済み** `wt[k, n]`（行優先）を受け取る。内側ループは
/// 「x のスカラーを放送して、連続な NR 要素に FMA する」形になり、
/// 内積のような水平和が一切不要になる。
///
/// `MR × NR` 個のアキュムレータを k ループの全体にわたってレジスタに保持するため、
/// 出力の読み書きは k ループの外で1回だけになり、FMA と load の比が `NR*MR : MR+NR`
/// まで改善する。イベント駆動側 (`linear_event`) と同じ転置レイアウトを使っており、
/// どちらか一方だけを優遇していない。
///
/// **アーキテクチャごとの選択**: x86_64 ではこのレジスタブロック版が速いが、
/// aarch64（Cortex-A57）では `linear_dense_panel` の方が速い（実測 2.1 → 2.7 GFLOPS）。
/// 密な側に不利な実装を選ばないよう、`linear_dense_t` が cfg で速い方へ振り分ける。
/// 3つの実行モードはすべて同じ関数を通るので、この選択は比較の公平性を損なわない。
#[inline]
pub fn linear_dense_t(
    x: &[f32],
    m: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    cnt: &mut MacCounters,
) {
    #[cfg(target_arch = "aarch64")]
    linear_dense_panel(x, m, k, wt, n, bias, out, cnt);
    #[cfg(not(target_arch = "aarch64"))]
    linear_dense_reg(x, m, k, wt, n, bias, out, cnt);
}

/// パネル分割版。出力パネルを L1 に収め、k を最内で回す。
/// aarch64 ではこちらが速い（レジスタ割り付けの都合と推測）。
pub fn linear_dense_panel(
    x: &[f32],
    m: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    cnt: &mut MacCounters,
) {
    const NB: usize = 256;
    for mi in 0..m {
        let o = &mut out[mi * n..mi * n + n];
        match bias {
            Some(b) => o.copy_from_slice(&b[..n]),
            None => o.fill(0.0),
        }
    }
    let mut m0 = 0;
    while m0 < m {
        let mr = MR.min(m - m0);
        let mut n0 = 0;
        while n0 < n {
            let nb = NB.min(n - n0);
            let mut rows: [&mut [f32]; MR] = [&mut [], &mut [], &mut [], &mut []];
            {
                let mut rest = &mut out[m0 * n..(m0 + mr) * n];
                for slot in rows.iter_mut().take(mr) {
                    let (head, tail) = rest.split_at_mut(n);
                    *slot = &mut head[n0..n0 + nb];
                    rest = tail;
                }
            }
            for ki in 0..k {
                let wrow = &wt[ki * n + n0..ki * n + n0 + nb];
                for (mi, row) in rows.iter_mut().enumerate().take(mr) {
                    // 密な実装なので、xv がゼロでもスキップはしない。
                    let xv = x[(m0 + mi) * k + ki];
                    for (oj, wj) in row[..nb].iter_mut().zip(wrow.iter()) {
                        *oj += xv * *wj;
                    }
                }
            }
            n0 += nb;
        }
        m0 += mr;
    }
    cnt.block_dense += (m * n * k) as u64;
    cnt.block_exec += (m * n * k) as u64;
}

pub fn linear_dense_reg(
    x: &[f32],
    m: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    cnt: &mut MacCounters,
) {
    let mut n0 = 0;
    while n0 < n {
        let nr = NR.min(n - n0);
        let mut m0 = 0;
        while m0 < m {
            let mr = MR.min(m - m0);
            let mut acc = [[0.0f32; NR]; MR];
            if mr == MR && nr == NR {
                // 主経路: MR も NR も固定長なので、アキュムレータがレジスタに載る。
                // （mi を実行時ループにすると LLVM が acc をスタックへ退避してしまう）
                let (a0, rest) = acc.split_at_mut(1);
                let (a1, rest) = rest.split_at_mut(1);
                let (a2, a3) = rest.split_at_mut(1);
                let (a0, a1, a2, a3) = (&mut a0[0], &mut a1[0], &mut a2[0], &mut a3[0]);
                let xr = &x[m0 * k..(m0 + MR) * k];
                for ki in 0..k {
                    let wrow = &wt[ki * n + n0..ki * n + n0 + NR];
                    // 密な実装なので、値がゼロでもスキップはしない（スパース性を
                    // 密側に持ち込むと比較の意味が失われる）。
                    let (x0, x1) = (xr[ki], xr[k + ki]);
                    let (x2, x3) = (xr[2 * k + ki], xr[3 * k + ki]);
                    for j in 0..NR {
                        let w = wrow[j];
                        a0[j] += x0 * w;
                        a1[j] += x1 * w;
                        a2[j] += x2 * w;
                        a3[j] += x3 * w;
                    }
                }
            } else {
                for ki in 0..k {
                    let wrow = &wt[ki * n + n0..ki * n + n0 + nr];
                    for mi in 0..mr {
                        let xv = x[(m0 + mi) * k + ki];
                        for (a, w) in acc[mi][..nr].iter_mut().zip(wrow.iter()) {
                            *a += xv * *w;
                        }
                    }
                }
            }
            for mi in 0..mr {
                let o = &mut out[(m0 + mi) * n + n0..(m0 + mi) * n + n0 + nr];
                match bias {
                    Some(b) => {
                        for ((oj, a), bj) in o.iter_mut().zip(&acc[mi][..nr]).zip(&b[n0..n0 + nr]) {
                            *oj = *a + *bj;
                        }
                    }
                    None => o.copy_from_slice(&acc[mi][..nr]),
                }
            }
            m0 += mr;
        }
        n0 += nr;
    }
    cnt.block_dense += (m * n * k) as u64;
    cnt.block_exec += (m * n * k) as u64;
}

/// イベント駆動の線形層: `out[n] = bias[n] + Σ_{i: s[i]=1} W[:, i]`
///
/// 入力は二値スパイクなので**乗算は一切不要**で、発火した入力ユニットに対応する
/// 重み列を足し合わせるだけになる。列アクセスを連続にするため、重みは
/// 転置済み `wt[in, out]`（行優先）で保持する。`fired` は発火した入力の添字列。
///
/// MACカウンタには `fired.len() * n` を計上する（密なら `k * n`）。
pub fn linear_event(
    fired: &[u32],
    wt: &[f32],
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    cnt: &mut MacCounters,
) {
    match bias {
        Some(b) => out[..n].copy_from_slice(&b[..n]),
        None => out[..n].fill(0.0),
    }
    let o = &mut out[..n];
    for &i in fired {
        let row = &wt[i as usize * n..i as usize * n + n];
        // 連続な n 要素の加算。密な側と同じイテレータ形に揃えてベクトル化させる。
        for (oj, rj) in o.iter_mut().zip(row.iter()) {
            *oj += *rj;
        }
    }
    cnt.block_dense += (n * k) as u64;
    cnt.block_exec += (n * fired.len()) as u64;
}

/// [in, out] の行優先に転置する（イベント駆動用の重み配置）。
pub fn transpose(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            t[j * n + i] = w[i * k + j];
        }
    }
    t
}

/// LayerNorm（PyTorch 既定の eps=1e-5、最終次元に対して）。
pub fn layer_norm(x: &[f32], out: &mut [f32], w: &[f32], b: &[f32], d: usize) {
    let mut mean = 0.0f32;
    for i in 0..d {
        mean += x[i];
    }
    mean /= d as f32;
    let mut var = 0.0f32;
    for i in 0..d {
        let t = x[i] - mean;
        var += t * t;
    }
    var /= d as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..d {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// ELU（PyTorch の F.elu、alpha=1）。
#[inline(always)]
pub fn elu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

/// 誤差関数（Abramowitz & Stegun 7.1.26、絶対誤差 1.5e-7）。
/// f32 の GELU を PyTorch の厳密版（erf ベース）と一致させるために使う。
#[inline(always)]
fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0f32 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

/// GELU（PyTorch 既定の厳密版）。
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// その場でsoftmaxを取る（数値安定化つき）。
pub fn softmax_inplace(v: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for &x in v.iter() {
        if x > mx {
            mx = x;
        }
    }
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    let inv = 1.0 / s;
    for x in v.iter_mut() {
        *x *= inv;
    }
}
