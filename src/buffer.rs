//! 【CoverageBuffer】
//!
//! 地磁気計測値が球面上に分布するようにデータを保持するバッファアルゴリズム

use nalgebra::Vector3;

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
    buffer: [Option<Entry>; BUFFER_SIZE],
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

    /// 有効なエントリをスロット順に参照する。
    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.buffer.iter().flatten()
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

    /// バッファをリセットする
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.crowded = (0, 0.0);
    }
}
