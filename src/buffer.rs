//! 【CoverageBuffer】
//! 
//! 地磁気計測値が球面上に分布するようにデータを保持するバッファアルゴリズム

use super::{Vector3, Matrix3x3};

/// バッファサイズ
pub const BUFFER_SIZE: usize = 16;

/// 角度閾値の理論計算値に掛ける係数（0〜1）
/// 
/// 1にするとバッファ内の点が球面上に均等に分布して
/// 信頼度が1になったときに新しい点が追加されなくなるから
/// 0.9程度にしておく（現実的には信頼度が1になることは無いけど）。
const ANGULAR_THR_COEF: f32 = 0.9;

/// バッファ内年齢閾値を調整する係数（小さすぎるとバッファが機能しないので注意）
/// 
/// バッファ内年齢の平均値×LIFETIME_THR_COEFを越えた点は削除される。
const LIFETIME_THR_COEF: f32 = 10.0;

/// 計測値の寿命を管理するための構造体
#[derive(Clone, Copy)]
pub struct Entry {
    pub point: Vector3<f32>,  // 地磁気計測値
    pub age: u16,  // バッファ内データの年齢（バッファ更新時に+1）
}

/// 分布度を維持するように地磁気計測値を保持するバッファ
pub struct CoverageBuffer {
    pub buffer: [Option<Entry>; BUFFER_SIZE],
    pub confidence_score: f32,  // データの信頼度（≒球状分布度）
}

impl CoverageBuffer {
    /// 空のバッファを作成する
    pub fn new() -> Self {
        Self {
            buffer: [None; BUFFER_SIZE],
            confidence_score: 0.0,
        }
    }

    /// バッファ内データの信頼度に応じて角度閾値を計算する
    /// 
    /// データの信頼度が低い（分布度が小さい）ときは、オフセット推定
    /// が上手くいっていない状態を考慮して角度閾値を小さくする。
    /// データの信頼度が高い（分布度が大きい）ときは、必要以上の
    /// バッファ更新が起こらないように角度閾値を大きくする。
    fn angular_threshold(&self) -> f32 {
        // N個の点が球面を等面積で分割するとした場合の、隣接する点同士の角度距離
        let theta = 2.0 * (1.0 - 2.0 / (BUFFER_SIZE as f32)).acos();

        // cos値で比較するために変換
        (ANGULAR_THR_COEF * theta * self.confidence_score).cos()
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

    /// 他点との距離が最小な点（一番密集している点）のインデックスを返す。
    fn most_crowded_idx(&self) -> usize {
        let mut most_crowded_idx = 0;  // 最密集点のインデックス
        let mut smallest_nearest_dist = f32::INFINITY; // 距離最小＝最密集

        for i in 0..BUFFER_SIZE {
            let Some(ei) = &self.buffer[i] else { continue };

            // 全ての組み合わせで点同士の距離を確認
            let mut min_dist = f32::INFINITY;
            for j in (i + 1)..BUFFER_SIZE {  // 同じ点同士を2回比較しないようにする
                if let Some(ej) = &self.buffer[j] {
                    let d = ei.point - ej.point;
                    let dist = d.x.abs() + d.y.abs() + d.z.abs();  // 一番シンプルな L1ノルム で見る
                    min_dist = min_dist.min(dist);
                }
            }
            // min_distが最も小さい点＝最も近い隣を持つ点＝一番密集している点
            if min_dist < smallest_nearest_dist {
                smallest_nearest_dist = min_dist;
                most_crowded_idx = i;
            }
        }

        most_crowded_idx
    }

    /// 新しい測定値を投入する
    /// * `point` : 生の地磁気計測値
    /// * `center`: 現時点でのオフセット推定値
    /// 
    /// 処理ステップ：
    /// 1. 既存の点との距離（角度）を見る。
    ///    a. バッファに空きがあれば無条件で追加。
    ///    b. バッファが満タンで、閾値内に既存点がある → スキップ
    ///    c. バッファが満タンで、閾値内に既存点が無い → 最近傍点を置き換え (ageインクリメント)
    /// 2. バッファが満タンの場合は寿命を越えた点を削除
    /// 3. 信頼度更新
    pub fn update(&mut self, point: Vector3<f32>, center: Vector3<f32>) {
        // --- Step.1 --- //
        // 既存の点との距離（角度）を見る
        // 角度比較はcenterで中心化・正規化してから行う
        let new_dir = (point - center).normalize();

        // 空きスロットの有無と、閾値内の点の有無を調べる。
        let mut nearest_flag = false;  // 閾値内の近い点があればtrue
        let mut empty_idx: Option<usize> = None;  // 空きスロットのインデックス
        for (i, slot) in self.buffer.iter().enumerate() {
            if let Some(entry) = slot {
                let existing_dir = (entry.point - center).normalize();
                let cos_angle = new_dir.dot(&existing_dir);  // 既存点との角度（のコサイン）
                if cos_angle > self.angular_threshold() {
                    nearest_flag = true;
                }
            } else {
                empty_idx = Some(i);
            }
        }

        if let Some(idx) = empty_idx {
            // バッファに空きがあるなら無条件で追加する
            self.insert(idx, point);
        } else {
            if nearest_flag == true {
                // バッファに空きがなく、角度閾値内に既存点があるならreturn。
                return;
            } else {
                // バッファに空きがなく角度閾値内に既存点も無ければ、再密集点を新しい点で置き換え。
                self.insert(self.most_crowded_idx(), point);    
            }
        }

        // --- Step.2 --- //
        // 寿命を越えた点を削除
        let lifetime = self.lifetime_threshold();
        for slot in self.buffer.iter_mut() {
            if let Some(entry) = slot {
                if entry.age > lifetime {
                    *slot = None;
                }
            }
        }

        // --- Step.3 --- //
        // 信頼度更新
        // 点の分布度を表す行列mを計算
        let mut m = Matrix3x3::<f32>::zeros();
        for slot in &self.buffer {
            if let Some(e) = slot {
                let dir = (e.point - center).normalize();
                m += dir * dir.transpose();
            }
        }

        // 条件数の考え方を基に、mの最小固有値からバッファ内データの信頼度を計算
        //
        // 一つの単位ベクトルuに対して、u uᵀ の対角成分はuのL2ノルムとなるから
        // trace(u uᵀ) == 1が成り立つ。mはこれをn点でやっているのでtrace(m)==nとなる。
        // つまり、trace(m)==nということは3つの固有値の合計がnということなので、
        // 最小固有値は必ずn/3以下になる。λ_min==n/3が成り立つのは、uが全方向に
        // 均等に分布している場合のみ。
        // 
        // * 均一に分布：lambda_min → n/3 (score → 1)
        // * 偏り大　　：lambda_min → 0   (score → 0)
        let eigenvalues = m.symmetric_eigenvalues();
        let lambda_min = eigenvalues.min();  // 最小固有値
        self.confidence_score = lambda_min * (3.0 / BUFFER_SIZE as f32);  // n/3で割って[0, 1]に正規化
    }

    /// バッファをリセットする
    pub fn reset(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.confidence_score = 0.0;
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
