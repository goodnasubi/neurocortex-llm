//! スパイキングLM（イベント駆動／密の2実装）と密なTransformerベースラインの推論。
//!
//! いずれも `src/neurocortex/lm.py` の PyTorch 実装と**同一の演算**を行う。
//! 重みは `results/weights/{spiking,dense}.npz` から読む（12.11.2節）。
//!
//! 3つの実行モード:
//!   * `SpikingMode::Event` … 発火したユニットの列だけを足し合わせる（本命）
//!   * `SpikingMode::Dense` … 同じスパイキングモデルを通常の密な行列積で
//!   * `DenseLm`            … `DenseBaseline`（softmaxアテンション + GELU）
//!
//! Event と Dense は同一のモデル・同一の重み・同一のスパイク列を通るので、
//! 差はイベント駆動性の有無だけになる。

use crate::linalg::*;
use crate::npz::{read_npz, Npz};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpikingMode {
    Event,
    Dense,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab: usize,
    pub max_len: usize,
    // ALIF のハイパーパラメータ（NeuronConfig と同じ既定値。beta のみ export 時 0.95）
    pub beta: f32,
    pub theta: f32,
    pub kappa: f32,
    pub lam: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            d_model: 192,
            n_layers: 3,
            n_heads: 4,
            vocab: 1014,
            max_len: 128,
            beta: 0.95,
            theta: 1.0,
            kappa: 0.5,
            lam: 0.9,
        }
    }
}

/// 線形層の重み。転置形 [in, out] を持つ（密・イベント駆動の双方がこれを使う）。
/// （転置形はモデル読み込み時に一度だけ作る。推論ループ内では作らない）
pub struct Linear {
    pub w: Vec<f32>,
    pub wt: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
}

impl Linear {
    fn load(z: &Npz, prefix: &str, with_bias: bool) -> Result<Linear, String> {
        let a = z.get(&format!("{prefix}.weight"))?;
        let (n_out, n_in) = (a.rows(), a.cols());
        let bias = if with_bias {
            Some(z.get(&format!("{prefix}.bias"))?.data.clone())
        } else {
            None
        };
        // 転置形 [in, out] は密・イベント駆動の**両方**が使う（読み込み時に一度だけ作る）
        let wt = transpose(&a.data, n_out, n_in);
        Ok(Linear { w: a.data.clone(), wt, bias, n_out, n_in })
    }
}

pub struct SpikingBlock {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv: Linear,
    pub out: Linear,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct SpikingLm {
    pub cfg: Config,
    pub embed: Vec<f32>,
    pub pos: Vec<f32>,
    pub blocks: Vec<SpikingBlock>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
    pub head: Linear,
}

/// 適応閾値LIF（`ALIFNeuron` と同じ漸化式）。膜電位はトークン位置方向に持ち越す。
struct Alif {
    v: Vec<f32>,
    a: Vec<f32>,
}

impl Alif {
    fn new(d: usize) -> Alif {
        Alif { v: vec![0.0; d], a: vec![0.0; d] }
    }
    fn reset(&mut self) {
        self.v.fill(0.0);
        self.a.fill(0.0);
    }
    /// 1トークン分を進め、二値スパイクを s に書き、発火した添字を fired に積む。
    #[inline]
    fn step(&mut self, cfg: &Config, x: &[f32], s: &mut [f32], fired: &mut Vec<u32>) {
        fired.clear();
        for i in 0..x.len() {
            let v = cfg.beta * self.v[i] + x[i];
            let thr = cfg.theta + self.a[i];
            let sp = if v - thr > 0.0 { 1.0f32 } else { 0.0 };
            self.v[i] = v - sp * thr;
            self.a[i] = cfg.lam * self.a[i] + cfg.kappa * sp;
            s[i] = sp;
            if sp != 0.0 {
                fired.push(i as u32);
            }
        }
    }
}

/// スパイク列の記録（PyTorch版との厳密一致テスト用。ベンチ時は無効）。
#[derive(Default)]
pub struct SpikeLog {
    pub enabled: bool,
    pub data: Vec<f32>,
}

impl SpikingLm {
    pub fn load<P: AsRef<Path>>(path: P, cfg: Config) -> Result<SpikingLm, String> {
        let z = read_npz(path)?;
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layers {
            let p = format!("blocks.{i}");
            blocks.push(SpikingBlock {
                ln1_w: z.get(&format!("{p}.ln1.weight"))?.data.clone(),
                ln1_b: z.get(&format!("{p}.ln1.bias"))?.data.clone(),
                qkv: Linear::load(&z, &format!("{p}.attn.qkv"), false)?,
                out: Linear::load(&z, &format!("{p}.attn.out"), false)?,
                ln2_w: z.get(&format!("{p}.ln2.weight"))?.data.clone(),
                ln2_b: z.get(&format!("{p}.ln2.bias"))?.data.clone(),
                fc1: Linear::load(&z, &format!("{p}.fc1"), true)?,
                fc2: Linear::load(&z, &format!("{p}.fc2"), true)?,
            });
        }
        Ok(SpikingLm {
            cfg,
            embed: z.get("embed.weight")?.data.clone(),
            pos: z.get("pos.weight")?.data.clone(),
            blocks,
            ln_f_w: z.get("ln_f.weight")?.data.clone(),
            ln_f_b: z.get("ln_f.bias")?.data.clone(),
            head: Linear::load(&z, "head", false)?,
        })
    }

    /// 1系列 [T] を通して logits [T, vocab] を返す。
    pub fn forward(
        &self,
        tokens: &[u32],
        mode: SpikingMode,
        cnt: &mut MacCounters,
        log: &mut SpikeLog,
        logits: &mut [f32],
    ) {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let t = tokens.len();
        let dh = d / cfg.n_heads;
        let dff = self.blocks[0].fc1.n_out;

        // 残差ストリーム（連続値のまま。加算のみ）
        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = &self.embed[tok as usize * d..tok as usize * d + d];
            let p = &self.pos[i * d..i * d + d];
            for j in 0..d {
                x[i * d + j] = e[j] + p[j];
            }
        }

        let mut h = vec![0.0f32; t * d];
        let mut spikes = vec![0.0f32; t * dff.max(d)];
        let mut qkv = vec![0.0f32; t * 3 * d];
        let mut yv = vec![0.0f32; t * d];
        let mut ffn = vec![0.0f32; t * dff];
        let mut resid = vec![0.0f32; t * d];
        let mut fired: Vec<Vec<u32>> = (0..t).map(|_| Vec::with_capacity(d.max(dff))).collect();
        let mut alif = Alif::new(dff.max(d));

        for blk in &self.blocks {
            // ---- アテンション経路: LN → ALIF → qkv ----
            for i in 0..t {
                layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &blk.ln1_w, &blk.ln1_b, d);
            }
            alif.reset();
            for i in 0..t {
                alif.step(&cfg, &h[i * d..i * d + d], &mut spikes[i * d..i * d + d], &mut fired[i]);
            }
            cnt.spikes += fired[..t].iter().map(|f| f.len() as u64).sum::<u64>();
            cnt.spike_slots += (t * d) as u64;
            if log.enabled {
                log.data.extend_from_slice(&spikes[..t * d]);
            }
            match mode {
                SpikingMode::Event => {
                    for i in 0..t {
                        linear_event(&fired[i], &blk.qkv.wt, 3 * d, d, None,
                                     &mut qkv[i * 3 * d..i * 3 * d + 3 * d], cnt);
                    }
                }
                SpikingMode::Dense => {
                    linear_dense_t(&spikes[..t * d], t, d, &blk.qkv.wt, 3 * d, None, &mut qkv, cnt);
                }
            }

            // ---- 因果的線形アテンション（重みを持たない。両モードとも密）----
            self.linear_attention(&qkv, t, dh, &mut yv, cnt);

            // ---- 出力射影（連続値入力なので常に密。両モードで同一）----
            linear_dense_t(&yv, t, d, &blk.out.wt, d, None, &mut resid, cnt);
            for j in 0..t * d {
                x[j] += resid[j];
            }

            // ---- FFN経路: LN → ALIF → fc1 → ALIF → fc2 ----
            for i in 0..t {
                layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &blk.ln2_w, &blk.ln2_b, d);
            }
            alif.reset();
            for i in 0..t {
                alif.step(&cfg, &h[i * d..i * d + d], &mut spikes[i * d..i * d + d], &mut fired[i]);
            }
            cnt.spikes += fired[..t].iter().map(|f| f.len() as u64).sum::<u64>();
            cnt.spike_slots += (t * d) as u64;
            if log.enabled {
                log.data.extend_from_slice(&spikes[..t * d]);
            }
            match mode {
                SpikingMode::Event => {
                    for i in 0..t {
                        linear_event(&fired[i], &blk.fc1.wt, dff, d, blk.fc1.bias.as_deref(),
                                     &mut ffn[i * dff..i * dff + dff], cnt);
                    }
                }
                SpikingMode::Dense => {
                    linear_dense_t(&spikes[..t * d], t, d, &blk.fc1.wt, dff,
                                 blk.fc1.bias.as_deref(), &mut ffn, cnt);
                }
            }
            alif.reset();
            for i in 0..t {
                alif.step(&cfg, &ffn[i * dff..i * dff + dff], &mut spikes[i * dff..i * dff + dff], &mut fired[i]);
            }
            cnt.spikes += fired[..t].iter().map(|f| f.len() as u64).sum::<u64>();
            cnt.spike_slots += (t * dff) as u64;
            if log.enabled {
                log.data.extend_from_slice(&spikes[..t * dff]);
            }
            match mode {
                SpikingMode::Event => {
                    for i in 0..t {
                        linear_event(&fired[i], &blk.fc2.wt, d, dff, blk.fc2.bias.as_deref(),
                                     &mut resid[i * d..i * d + d], cnt);
                    }
                }
                SpikingMode::Dense => {
                    linear_dense_t(&spikes[..t * dff], t, dff, &blk.fc2.wt, d,
                                 blk.fc2.bias.as_deref(), &mut resid, cnt);
                }
            }
            for j in 0..t * d {
                x[j] += resid[j];
            }
        }

        // ---- 最終LN と出力ヘッド（語彙1014。配置に依らず密）----
        for i in 0..t {
            layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &self.ln_f_w, &self.ln_f_b, d);
        }
        let mut hc = MacCounters::default();
        linear_dense_t(&h[..t * d], t, d, &self.head.wt, cfg.vocab, None, logits, &mut hc);
        cnt.head += hc.block_exec;
    }

    /// 因果的線形アテンション。状態 S, z を位置方向に持ち越す（attention.py と同じ）。
    fn linear_attention(&self, qkv: &[f32], t: usize, dh: usize, y: &mut [f32], cnt: &mut MacCounters) {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let nh = cfg.n_heads;
        let mut s = vec![0.0f32; dh * dh];
        let mut z = vec![0.0f32; dh];
        let mut q = vec![0.0f32; dh];
        let mut k = vec![0.0f32; dh];
        for hh in 0..nh {
            s.fill(0.0);
            z.fill(0.0);
            for i in 0..t {
                let base = i * 3 * d + hh * dh;
                // φ(x) = elu(x) + 1 で非負化
                for j in 0..dh {
                    q[j] = elu(qkv[base + j]) + 1.0;
                    k[j] = elu(qkv[base + d + j]) + 1.0;
                }
                let v = &qkv[base + 2 * d..base + 2 * d + dh];
                // S += k ⊗ v, z += k
                for a in 0..dh {
                    let ka = k[a];
                    let row = &mut s[a * dh..a * dh + dh];
                    for b in 0..dh {
                        row[b] += ka * v[b];
                    }
                    z[a] += ka;
                }
                // num = q·S, den = q·z
                let out = &mut y[i * d + hh * dh..i * d + hh * dh + dh];
                out.fill(0.0);
                for a in 0..dh {
                    let qa = q[a];
                    let row = &s[a * dh..a * dh + dh];
                    for b in 0..dh {
                        out[b] += qa * row[b];
                    }
                }
                let mut den = 0.0f32;
                for a in 0..dh {
                    den += q[a] * z[a];
                }
                let inv = 1.0 / (den + 1e-6);
                for b in 0..dh {
                    out[b] *= inv;
                }
            }
        }
        // k⊗v, q·S, q·z で 3 * dh^2 相当（12.6.3節の別枠と同じ計上）
        cnt.attn_state += (t * nh * 3 * dh * dh) as u64;
    }
}

// ===================== 密なベースライン（DenseBaseline） =====================

pub struct DenseBlock {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv: Linear,
    pub proj: Linear,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct DenseLm {
    pub cfg: Config,
    pub embed: Vec<f32>,
    pub pos: Vec<f32>,
    pub blocks: Vec<DenseBlock>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
    pub head: Linear,
}

impl DenseLm {
    pub fn load<P: AsRef<Path>>(path: P, cfg: Config) -> Result<DenseLm, String> {
        let z = read_npz(path)?;
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layers {
            let p = format!("blocks.{i}");
            blocks.push(DenseBlock {
                ln1_w: z.get(&format!("{p}.ln1.weight"))?.data.clone(),
                ln1_b: z.get(&format!("{p}.ln1.bias"))?.data.clone(),
                qkv: Linear::load(&z, &format!("{p}.qkv"), false)?,
                proj: Linear::load(&z, &format!("{p}.proj"), false)?,
                ln2_w: z.get(&format!("{p}.ln2.weight"))?.data.clone(),
                ln2_b: z.get(&format!("{p}.ln2.bias"))?.data.clone(),
                fc1: Linear::load(&z, &format!("{p}.fc1"), true)?,
                fc2: Linear::load(&z, &format!("{p}.fc2"), true)?,
            });
        }
        Ok(DenseLm {
            cfg,
            embed: z.get("embed.weight")?.data.clone(),
            pos: z.get("pos.weight")?.data.clone(),
            blocks,
            ln_f_w: z.get("ln_f.weight")?.data.clone(),
            ln_f_b: z.get("ln_f.bias")?.data.clone(),
            head: Linear::load(&z, "head", false)?,
        })
    }

    pub fn forward(&self, tokens: &[u32], cnt: &mut MacCounters, logits: &mut [f32]) {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let t = tokens.len();
        let nh = cfg.n_heads;
        let dh = d / nh;
        let dff = self.blocks[0].fc1.n_out;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = &self.embed[tok as usize * d..tok as usize * d + d];
            let p = &self.pos[i * d..i * d + d];
            for j in 0..d {
                x[i * d + j] = e[j] + p[j];
            }
        }
        let mut h = vec![0.0f32; t * d];
        let mut qkv = vec![0.0f32; t * 3 * d];
        let mut y = vec![0.0f32; t * d];
        let mut resid = vec![0.0f32; t * d];
        let mut ffn = vec![0.0f32; t * dff];
        let mut att = vec![0.0f32; t];

        for blk in &self.blocks {
            for i in 0..t {
                layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &blk.ln1_w, &blk.ln1_b, d);
            }
            linear_dense_t(&h[..t * d], t, d, &blk.qkv.wt, 3 * d, None, &mut qkv, cnt);
            // 因果的softmaxアテンション
            for hh in 0..nh {
                for i in 0..t {
                    let qi = &qkv[i * 3 * d + hh * dh..i * 3 * d + hh * dh + dh];
                    for j in 0..=i {
                        let kj = &qkv[j * 3 * d + d + hh * dh..j * 3 * d + d + hh * dh + dh];
                        att[j] = dot(qi, kj) * scale;
                    }
                    softmax_inplace(&mut att[..i + 1]);
                    let o = &mut y[i * d + hh * dh..i * d + hh * dh + dh];
                    o.fill(0.0);
                    for j in 0..=i {
                        let a = att[j];
                        let vj = &qkv[j * 3 * d + 2 * d + hh * dh..j * 3 * d + 2 * d + hh * dh + dh];
                        for b in 0..dh {
                            o[b] += a * vj[b];
                        }
                    }
                    cnt.attn_state += (2 * (i + 1) * dh) as u64;
                }
            }
            linear_dense_t(&y, t, d, &blk.proj.wt, d, None, &mut resid, cnt);
            for j in 0..t * d {
                x[j] += resid[j];
            }
            for i in 0..t {
                layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &blk.ln2_w, &blk.ln2_b, d);
            }
            linear_dense_t(&h[..t * d], t, d, &blk.fc1.wt, dff, blk.fc1.bias.as_deref(), &mut ffn, cnt);
            for v in ffn.iter_mut() {
                *v = gelu(*v);
            }
            linear_dense_t(&ffn, t, dff, &blk.fc2.wt, d, blk.fc2.bias.as_deref(), &mut resid, cnt);
            for j in 0..t * d {
                x[j] += resid[j];
            }
        }
        for i in 0..t {
            layer_norm(&x[i * d..i * d + d], &mut h[i * d..i * d + d], &self.ln_f_w, &self.ln_f_b, d);
        }
        let mut hc = MacCounters::default();
        linear_dense_t(&h[..t * d], t, d, &self.head.wt, cfg.vocab, None, logits, &mut hc);
        cnt.head += hc.block_exec;
    }
}
