//! 【CoverageBuffer】
//!
//! 地磁気計測値が球面上に分布するようにデータを保持するバッファアルゴリズム

use super::{Matrix3x3, Vector3};

/// バッファサイズ
pub const BUFFER_SIZE: usize = 16;

/// バッファ内年齢閾値を調整する係数（小さすぎるとバッファが機能しないので注意）
///
/// バッファ内年齢の平均値×LIFETIME_THR_COEFを越えた点は削除される。
const LIFETIME_THR_COEF: f32 = 10.0;

/// 計測値の寿命を管理するための構造体
#[derive(Clone, Copy)]
pub struct Entry {
    pub point: Vector3<f32>, // 地磁気計測値
    pub age: u16,            // バッファ内データの年齢（新点追加時に+1）
}

/// 空間上の均等な分布を維持するように地磁気計測値を保持するバッファ
pub struct CoverageBuffer {
    pub buffer: [Option<Entry>; BUFFER_SIZE],
    crowded: (usize, f32), // (最密集点のインデックス, 最密集ペアの二乗距離)
}

impl CoverageBuffer {
    /// 空のバッファを作成する
    pub fn new() -> Self {
        Self {
            buffer: [None; BUFFER_SIZE],
            crowded: (0, 0.0),
        }
    }

    /// 寿命閾値を計算
    ///
    /// バッファ内年齢の平均値×AGE_THRESHOLDを寿命閾値とする。
    fn lifetime_threshold(&self) -> u16 {
        // 平均値計算（中央値は計算コストが高いから平均値にした）
        let mut mean = 0.0;
        let mut n: usize = 0;
        for slot in &self.buffer {
            if let Some(entry) = slot {
                mean += entry.age as f32;
                n += 1;
            }
        }
        if n == 0 {
            return u16::MAX;
        }
        mean /= n as f32;

        // 寿命閾値
        (mean * LIFETIME_THR_COEF) as u16
    }

    /// 指定したインデックスに点を挿入し、挿入した点以外のageをインクリメントする。
    ///
    /// 指定したインデックスに点が存在する場合は上書きする。
    fn insert(&mut self, idx: usize, point: Vector3<f32>) {
        for (i, slot) in self.buffer.iter_mut().enumerate() {
            if i == idx {
                *slot = Some(Entry { point, age: 0 });
            } else if let Some(entry) = slot {
                entry.age = entry.age.saturating_add(1);
            }
        }
    }

    /// 最密集点のインデックスと二乗距離を計算する。
    ///
    /// 最密ペアのうち、2番目の近傍点が近い方を選ぶ。
    /// 全ペアを調べるので計算量はO(N²)。
    fn search_crowded_point(&mut self) {
        // 1. 互いの距離が最短となるペアを選ぶ。
        let mut closest_pair = None;
        let mut min_sq = f32::INFINITY;
        for (i, slot) in self.buffer.iter().enumerate() {
            let Some(entry) = slot else { continue };
            for (j, other) in self.buffer.iter().enumerate().skip(i + 1) {
                let Some(other) = other else { continue };
                let distance_sq = (entry.point - other.point).norm_squared();
                if distance_sq < min_sq {
                    min_sq = distance_sq;
                    closest_pair = Some((i, j));
                }
            }
        }

        // 2点未満ならペアがないため二乗距離は0。インデックスはなんでもいいので0にしておく。
        self.crowded = (0, 0.0);
        let Some((i, j)) = closest_pair else { return };
        let a = self.buffer[i].unwrap();
        let b = self.buffer[j].unwrap();

        // 2. 選んだペアのうち、他の点までの距離がより近い方を最密集点として選ぶ。
        let mut second_a = f32::INFINITY;
        let mut second_b = f32::INFINITY;
        for (k, slot) in self.buffer.iter().enumerate() {
            if k == i || k == j {
                continue;
            }
            let Some(entry) = slot else { continue };
            second_a = second_a.min((a.point - entry.point).norm_squared());
            second_b = second_b.min((b.point - entry.point).norm_squared());
        }

        // 3. 最密集ペアのうち周囲の点により近い方を置き換え候補にする
        let crowded_idx = if second_b < second_a { j } else { i };
        self.crowded = (crowded_idx, min_sq);
    }

    /// 新しい地磁気計測値を投入する。
    /// 
    /// 既存点が最密ペア距離の半分以内ならスキップする。
    pub fn update(&mut self, point: Vector3<f32>) {
        // 1. 新点の最近傍距離と、空きスロットを調べる。
        let mut dn_sq = f32::INFINITY;  // 新点と既存点の二乗距離の最小値。空なら∞。
        let mut empty_idx = None;  // 追加先に使える空きスロット
        for (i, slot) in self.buffer.iter().enumerate() {
            if let Some(entry) = slot {
                dn_sq = dn_sq.min((point - entry.point).norm_squared());
            } else {
                empty_idx = Some(i);
            }
        }
        // 空きスロットの有無にかかわらず、既存点に近ければスキップする。
        // 既存点が2点未満ならdmin_sqは0なので、同一点だけが弾かれる。
        if dn_sq <= 0.25 * self.crowded.1 {  // 二乗距離なので係数は0.5^2=0.25
            return;
        }

        // 2. 空きがあればそのまま追加し、満杯なら最密集点を置き換える。
        self.insert(empty_idx.unwrap_or(self.crowded.0), point);

        // 3. 寿命閾値を超えた点を削除
        let thr = self.lifetime_threshold();
        for slot in &mut self.buffer {
            if slot.as_ref().is_some_and(|entry| entry.age > thr) {
                *slot = None;
            }
        }

        // 4. 最密集点を再探索
        self.search_crowded_point();
    }

    /// バッファ内の点群の重心を返す。空なら `None`。
    pub fn centroid(&self) -> Option<Vector3<f32>> {
        let mut sum = Vector3::zeros();
        let mut n = 0;
        for entry in self.buffer.iter().flatten() {
            sum += entry.point;
            n += 1;
        }
        if n == 0 {
            return None;
        }
        Some(sum / n as f32)
    }

    /// 指定した中心 `center` を基準に、現在の点群の信頼度を計算する。
    pub fn confidence_score(&self, center: Vector3<f32>) -> f32 {
        let mut m = Matrix3x3::<f32>::zeros(); // 分散共分散行列
        let mut n = 0;
        let mut sum_d2 = 0.0; // Σ|d|²（= tr(M)）
        let mut sum_d4 = 0.0; // Σ|d|⁴
        for entry in self.buffer.iter().flatten() {
            let d = entry.point - center;
            let d2 = d.norm_squared();
            m += d * d.transpose();
            sum_d2 += d2;
            sum_d4 += d2 * d2;
            n += 1;
        }

        // 点が2つ未満、または全点が中心と一致する場合は信頼度を0にする。
        if n < 2 || sum_d2 <= 0.0 {
            return 0.0;
        }

        // 殻らしさfを計算
        // 球面上なら k = 1、3次元ガウスの塊なら k = 5/3
        let k = (n as f32) * sum_d4 / (sum_d2 * sum_d2);
        let f = ((5.0 / 3.0 - k) * 1.5).clamp(0.0, 1.0);
        if f == 0.0 {
            // 積が0になるので固有値計算を省略
            return 0.0;
        }

        // 充足度
        let fill_rate = n as f32 / BUFFER_SIZE as f32;

        // 等方性：2次モーメントで「丸さ」を見る
        let lambda_min = m.symmetric_eigenvalues().min();
        // 旧式：全球の等方分布を満点とする。
        // let isotropy = 3.0 * lambda_min / sum_d2;

        // 半球の一様分布では、重心まわりの3λ_min/tr(M)が1/3になる。
        // これを満点に換算し、それ以上の等方性は1で頭打ちにする。
        // 上限を設けるのは等方性だけで、充足度と殻らしさは引き続き掛ける。
        let isotropy = (9.0 * lambda_min / sum_d2).min(1.0);

        (isotropy * fill_rate * f).clamp(0.0, 1.0)
    }

    /// バッファをリセットする
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.crowded = (0, 0.0);
    }

    /// バッファ内の有効な点の数を返す
    pub fn count(&self) -> usize {
        let mut n = 0;
        for slot in &self.buffer {
            if slot.is_some() {
                n += 1;
            }
        }
        n
    }

    /// バッファ内の点を配列にコピーして返す（先頭count()個が有効）
    ///
    /// main.rsでのバッファデータ出力用
    pub fn points(&self) -> [Vector3<f32>; BUFFER_SIZE] {
        let mut out = [Vector3::<f32>::zeros(); BUFFER_SIZE];
        let mut n = 0;
        for slot in &self.buffer {
            if let Some(e) = slot {
                out[n] = e.point;
                n += 1;
            }
        }
        out
    }
}
