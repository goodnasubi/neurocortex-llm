//! numpy の `.npz` / `.npy` を標準ライブラリだけで読むための最小実装。
//!
//! `np.savez` は ZIP_STORED（無圧縮）で書き出すため、中央ディレクトリを読めば
//! 各エントリの生データがそのまま取り出せる。外部クレートを使わないのは、
//! Jetson（aarch64・オフライン気味）でのビルドを確実にするためである。

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// 1つの配列（f32 に正規化して保持する）。
#[derive(Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Array {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// [rows, cols] として扱う（1次元なら [1, n]）。
    pub fn rows(&self) -> usize {
        if self.shape.len() >= 2 { self.shape[0] } else { 1 }
    }
    pub fn cols(&self) -> usize {
        if self.shape.len() >= 2 { self.shape[1] } else { self.shape.first().copied().unwrap_or(0) }
    }
}

pub struct Npz(pub HashMap<String, Array>);

impl Npz {
    pub fn get(&self, key: &str) -> Result<&Array, String> {
        self.0.get(key).ok_or_else(|| format!(".npz に {key} が無い"))
    }
    pub fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
}

fn u16le(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}
fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// `.npz`（ZIP_STORED）を読む。中央ディレクトリ経由なのでデータ記述子付きでも問題ない。
pub fn read_npz<P: AsRef<Path>>(path: P) -> Result<Npz, String> {
    let buf = fs::read(path.as_ref()).map_err(|e| format!("{:?}: {e}", path.as_ref()))?;
    // EOCD（中央ディレクトリ終端レコード）を末尾から探す
    let mut eocd = None;
    let start = buf.len().saturating_sub(66_000);
    for i in (start..buf.len().saturating_sub(21)).rev() {
        if buf[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.ok_or("ZIPのEOCDが見つからない")?;
    let n_entries = u16le(&buf, eocd + 10);
    let mut off = u32le(&buf, eocd + 16); // 中央ディレクトリ開始位置

    let mut map = HashMap::new();
    for _ in 0..n_entries {
        if buf[off..off + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err("中央ディレクトリの署名が不正".into());
        }
        let method = u16le(&buf, off + 10);
        let csize = u32le(&buf, off + 20);
        let name_len = u16le(&buf, off + 28);
        let extra_len = u16le(&buf, off + 30);
        let comment_len = u16le(&buf, off + 32);
        let local_off = u32le(&buf, off + 42);
        let name = String::from_utf8_lossy(&buf[off + 46..off + 46 + name_len]).to_string();
        off += 46 + name_len + extra_len + comment_len;
        if method != 0 {
            return Err(format!("{name}: 圧縮された .npz は非対応（np.savez を使うこと）"));
        }
        // ローカルヘッダを読んでデータ開始位置を求める
        let ln = u16le(&buf, local_off + 26);
        let le = u16le(&buf, local_off + 28);
        let data_off = local_off + 30 + ln + le;
        let arr = parse_npy(&buf[data_off..data_off + csize])
            .map_err(|e| format!("{name}: {e}"))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        map.insert(key, arr);
    }
    Ok(Npz(map))
}

/// `.npy`（v1/v2, little-endian, C順）を f32 配列として読む。f4 / f8 / i8(int64) に対応。
pub fn parse_npy(b: &[u8]) -> Result<Array, String> {
    if b.len() < 10 || &b[0..6] != b"\x93NUMPY" {
        return Err("npyのマジックが不正".into());
    }
    let major = b[6];
    let (hdr_len, hdr_start) = if major == 1 {
        (u16le(b, 8), 10)
    } else {
        (u32le(b, 8), 12)
    };
    let header = String::from_utf8_lossy(&b[hdr_start..hdr_start + hdr_len]).to_string();
    let body = &b[hdr_start + hdr_len..];

    let descr = extract(&header, "'descr':").ok_or("descr が無い")?;
    if header.contains("'fortran_order': True") {
        return Err("fortran_order は非対応".into());
    }
    let shape = parse_shape(&header)?;
    let n: usize = shape.iter().product();

    let data = match descr.as_str() {
        "<f4" | "|f4" => (0..n)
            .map(|i| f32::from_le_bytes([body[4 * i], body[4 * i + 1], body[4 * i + 2], body[4 * i + 3]]))
            .collect(),
        "<f8" => (0..n)
            .map(|i| {
                let mut a = [0u8; 8];
                a.copy_from_slice(&body[8 * i..8 * i + 8]);
                f64::from_le_bytes(a) as f32
            })
            .collect(),
        "<i8" => (0..n)
            .map(|i| {
                let mut a = [0u8; 8];
                a.copy_from_slice(&body[8 * i..8 * i + 8]);
                i64::from_le_bytes(a) as f32
            })
            .collect(),
        "<i4" => (0..n)
            .map(|i| {
                i32::from_le_bytes([body[4 * i], body[4 * i + 1], body[4 * i + 2], body[4 * i + 3]]) as f32
            })
            .collect(),
        other => return Err(format!("非対応のdtype: {other}")),
    };
    Ok(Array { shape, data })
}

fn extract(header: &str, key: &str) -> Option<String> {
    let i = header.find(key)? + key.len();
    let rest = &header[i..];
    let a = rest.find('\'')? + 1;
    let b = rest[a..].find('\'')? + a;
    Some(rest[a..b].to_string())
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let i = header.find("'shape':").ok_or("shape が無い")? + 8;
    let rest = &header[i..];
    let a = rest.find('(').ok_or("shape の括弧が無い")? + 1;
    let b = rest.find(')').ok_or("shape の括弧が無い")?;
    let inner = &rest[a..b];
    let mut out = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<usize>().map_err(|e| format!("shape解析失敗: {e}"))?);
    }
    Ok(out)
}
