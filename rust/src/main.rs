//! Rust推論ベンチ本体（12.11.2節「ステップ1以降」）。
//!
//! 使い方:
//!   neurocortex-rs verify --ref results/rust/reference.npz --weights-dir results/weights
//!   neurocortex-rs bench  --mode event|spiking-dense|dense --iters 30
//!
//! 公平性のため、3モードとも同じバイナリ・同じ最適化設定・同じ入力トークン列で走らせる。

mod linalg;
mod model;
mod npz;

use linalg::MacCounters;
use model::{Config, DenseLm, SpikeLog, SpikingLm, SpikingMode};
use std::time::Instant;

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn argn<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    match args.windows(2).find(|w| w[0] == key) {
        Some(w) => w[1].parse().unwrap_or(default),
        None => default,
    }
}

/// ベンチ用の決定的なトークン列（PyTorch 側の `torch.randint(generator=0)` と
/// 同じものを使うため、参照ファイルから読み込む。無ければ線形合同法で生成する）。
fn fallback_tokens(b: usize, t: usize, vocab: usize) -> Vec<u32> {
    let mut s: u64 = 12345;
    (0..b * t)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % vocab as u64) as u32
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_default();
    match cmd.as_str() {
        "verify" => verify(&args),
        "bench" => bench(&args),
        "calib" => calib(&args),
        _ => {
            eprintln!("使い方: neurocortex-rs [verify|bench] ...");
            std::process::exit(2);
        }
    }
}

fn load_config(args: &[String]) -> Config {
    Config {
        d_model: argn(args, "--d-model", 192),
        n_layers: argn(args, "--n-layers", 3),
        n_heads: argn(args, "--n-heads", 4),
        vocab: argn(args, "--vocab", 1014),
        max_len: argn(args, "--max-len", 128),
        beta: argn(args, "--beta", 0.95f32),
        ..Config::default()
    }
}

// ============================ 正当性の検証 ============================

fn verify(args: &[String]) {
    let refpath = arg(args, "--ref", "results/rust/reference.npz");
    let wdir = arg(args, "--weights-dir", "results/weights");
    let z = npz::read_npz(&refpath).unwrap_or_else(|e| {
        eprintln!("参照ファイルが読めない: {e}");
        std::process::exit(1)
    });
    let cfg = load_config(args);
    let toks = z.get("tokens").expect("tokens");
    let (b, t) = (toks.rows(), toks.cols());
    let tokens: Vec<u32> = toks.data.iter().map(|&v| v as u32).collect();

    let mut fail = 0;

    // --- スパイキング: イベント駆動と密、両方を照合する ---
    for (label, mode) in [("event", SpikingMode::Event), ("spiking-dense", SpikingMode::Dense)] {
        let m = SpikingLm::load(format!("{wdir}/spiking.npz"), cfg).expect("spiking.npz");
        let mut logits = vec![0.0f32; b * t * cfg.vocab];
        let mut log = SpikeLog { enabled: true, data: Vec::new() };
        let mut cnt = MacCounters::default();
        for i in 0..b {
            m.forward(
                &tokens[i * t..i * t + t],
                mode,
                &mut cnt,
                &mut log,
                &mut logits[i * t * cfg.vocab..(i + 1) * t * cfg.vocab],
            );
        }
        // スパイク列は二値なので厳密一致するはず。
        // ただし閾値判定は和の順序に敏感で、float32 と float64 の PyTorch 同士でも
        // ごく一部のユニットで発火が反転する。そこで両方の参照と照合し、
        // **いずれかと厳密一致**すれば合格とする。
        let mut best_ok = false;
        for suffix in ["", "_f64"] {
            let rs = z.get(&format!("spikes{suffix}")).expect("spikes");
            let mismatch = log.data.iter().zip(rs.data.iter()).filter(|(a, b)| a != b).count();
            let lg = z.get(&format!("spiking_logits{suffix}")).expect("spiking_logits");
            let (maxabs, maxrel) = diff(&logits, &lg.data);
            let ok = mismatch == 0 && maxrel < 1e-4 && logits.iter().all(|v| v.is_finite());
            best_ok |= ok;
            println!(
                "[{label}] 参照 torch{} : スパイク不一致 {}/{} / logits 最大絶対誤差 {:.3e} 最大相対誤差 {:.3e}{}",
                if suffix.is_empty() { "-f32" } else { "-f64" },
                mismatch,
                log.data.len(),
                maxabs,
                maxrel,
                if ok { " ← 厳密一致" } else { "" }
            );
        }
        println!("[{label}] → {}", if best_ok { "合格" } else { "不合格" });
        if !best_ok {
            fail += 1;
        }
    }

    // --- 密なベースライン ---
    {
        let m = DenseLm::load(format!("{wdir}/dense.npz"), cfg).expect("dense.npz");
        let mut logits = vec![0.0f32; b * t * cfg.vocab];
        let mut cnt = MacCounters::default();
        for i in 0..b {
            m.forward(
                &tokens[i * t..i * t + t],
                &mut cnt,
                &mut logits[i * t * cfg.vocab..(i + 1) * t * cfg.vocab],
            );
        }
        let mut ok_any = false;
        for suffix in ["", "_f64"] {
            let lg = z.get(&format!("dense_logits{suffix}")).expect("dense_logits");
            let (maxabs, maxrel) = diff(&logits, &lg.data);
            let ok = maxrel < 1e-4 && logits.iter().all(|v| v.is_finite());
            ok_any |= ok;
            println!(
                "[dense] 参照 torch{} : logits 最大絶対誤差 {:.3e} 最大相対誤差 {:.3e}",
                if suffix.is_empty() { "-f32" } else { "-f64" }, maxabs, maxrel
            );
        }
        println!("[dense] → {}", if ok_any { "合格" } else { "不合格" });
        if !ok_any {
            fail += 1;
        }
    }

    // --- イベント駆動と密なスパイキングの相互一致（同一モデルなので）---
    println!("{}", if fail == 0 { "検証: 全項目合格" } else { "検証: 不合格あり" });
    if fail > 0 {
        std::process::exit(1);
    }
}

/// 最大絶対誤差と、最大絶対値で正規化した相対誤差を返す。
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut maxabs = 0.0f32;
    let mut scale = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        maxabs = maxabs.max((a[i] - b[i]).abs());
        scale = scale.max(b[i].abs());
    }
    (maxabs, if scale > 0.0 { maxabs / scale } else { maxabs })
}

// ==================== 密な行列積の素の性能（校正用） ====================

/// 自作の密なカーネルが何GFLOPS出ているかを測る。
/// PyTorch(oneDNN/BLAS) の同じ形の行列積と比べることで、「密な側の実装が
/// 手抜きでないか」を数値で開示するための校正値（12.11.2節 公平性）。
fn calib(args: &[String]) {
    let m: usize = argn(args, "--m", 64);
    let k: usize = argn(args, "--k", 192);
    let n: usize = argn(args, "--n", 768);
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let wt: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
    let mut out = vec![0.0f32; m * n];
    let mut cnt = MacCounters::default();
    let mut reps = 0u64;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 2.0 {
        for _ in 0..200 {
            linalg::linear_dense_t(&x, m, k, &wt, n, None, &mut out, &mut cnt);
        }
        reps += 200;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "CALIB rust linear_dense_t {m}x{k}x{n}: {:.2} GFLOPS (1 thread, checksum={:.3})",
        2.0 * (m * k * n) as f64 * reps as f64 / dt / 1e9,
        out[0]
    );
}

// ============================ ベンチマーク ============================

fn bench(args: &[String]) {
    let mode = arg(args, "--mode", "event");
    let wdir = arg(args, "--weights-dir", "results/weights");
    let cfg = load_config(args);
    let t: usize = argn(args, "--seq-len", 64);
    let b: usize = argn(args, "--batch-size", 8);
    let iters: usize = argn(args, "--iters", 30);
    let warmup: usize = argn(args, "--warmup", 3);

    // 入力トークン列は3モードで完全に同一にする（公平性）
    let tokens = match npz::read_npz(arg(args, "--tokens", "results/rust/reference.npz")) {
        Ok(z) => match z.get("tokens") {
            Ok(a) if a.rows() == b && a.cols() == t => a.data.iter().map(|&v| v as u32).collect(),
            _ => fallback_tokens(b, t, cfg.vocab),
        },
        Err(_) => fallback_tokens(b, t, cfg.vocab),
    };

    let mut logits = vec![0.0f32; t * cfg.vocab];
    let mut log = SpikeLog::default();
    let mut cnt = MacCounters::default();

    enum M {
        S(SpikingLm, SpikingMode),
        D(DenseLm),
    }
    let m = match mode.as_str() {
        "event" => M::S(
            SpikingLm::load(format!("{wdir}/spiking.npz"), cfg).unwrap(),
            SpikingMode::Event,
        ),
        "spiking-dense" => M::S(
            SpikingLm::load(format!("{wdir}/spiking.npz"), cfg).unwrap(),
            SpikingMode::Dense,
        ),
        "dense" => M::D(DenseLm::load(format!("{wdir}/dense.npz"), cfg).unwrap()),
        other => {
            eprintln!("未知のモード: {other}");
            std::process::exit(2)
        }
    };

    let run = |cnt: &mut MacCounters, logits: &mut [f32], log: &mut SpikeLog| match &m {
        M::S(model, mode) => {
            for i in 0..b {
                model.forward(&tokens[i * t..i * t + t], *mode, cnt, log, logits);
            }
        }
        M::D(model) => {
            for i in 0..b {
                model.forward(&tokens[i * t..i * t + t], cnt, logits);
            }
        }
    };

    for _ in 0..warmup {
        run(&mut cnt, &mut logits, &mut log);
    }
    // カウンタは計測区間のものだけを残す
    cnt = MacCounters::default();
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut cnt, &mut logits, &mut log);
    }
    let dt = t0.elapsed().as_secs_f64();

    let n_tokens = (iters * b * t) as f64;
    let finite = logits.iter().all(|v| v.is_finite());
    let per = |x: u64| x as f64 / n_tokens;
    let block_dense = per(cnt.block_dense);
    let block_exec = per(cnt.block_exec);
    let head = per(cnt.head);
    let attn = per(cnt.attn_state);
    let total_dense = block_dense + head + attn;
    let total_exec = block_exec + head + attn;
    let rate = if cnt.spike_slots > 0 {
        cnt.spikes as f64 / cnt.spike_slots as f64
    } else {
        0.0
    };

    println!(
        "RESULT {{\"impl\":\"rust\",\"mode\":\"{mode}\",\"seq_len\":{t},\"batch_size\":{b},\
\"iters\":{iters},\"tokens\":{},\"sec\":{:.4},\"tokens_per_sec\":{:.1},\"finite\":{finite},\
\"firing_rate\":{:.6},\"mac_per_token\":{{\"block_dense\":{:.0},\"block_exec\":{:.0},\
\"head\":{:.0},\"attn_state\":{:.0},\"total_dense\":{:.0},\"total_exec\":{:.0}}},\
\"skippable_block\":{:.6},\"skippable_total\":{:.6}}}",
        n_tokens as u64,
        dt,
        n_tokens / dt,
        rate,
        block_dense,
        block_exec,
        head,
        attn,
        total_dense,
        total_exec,
        if block_dense > 0.0 { 1.0 - block_exec / block_dense } else { 0.0 },
        if total_dense > 0.0 { 1.0 - total_exec / total_dense } else { 0.0 },
    );
}
