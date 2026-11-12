//! 較正計算の前後二段階で評価する。
//!
//! 事前評価：点群が中心を決められるだけ広がっているか（等方性 × 充足度）
//! 事後評価：フィットした中心に対して点群が中が空洞の殻状になっているか（殻らしさ）

use super::{Vector3, Matrix3x3};
use super::buffer::BUFFER_SIZE;

/// 較正パラメータ計算の事前評価に用いる信頼度
/// 
/// バッファの充填度 & 点の等方性で評価する。
pub fn pre_score(points: &[Vector3<f32>]) -> f32 {
    // 楕円体（自由度9）は9点以下だと全点をぴったり通ってしまい事後評価の
    // 殻らしさが必ず1になる。これでは評価の意味が無いので、10点未満は
    // 評価しない（1点だけのときに起こるゼロ除算もここで弾ける）。
    if points.len() < 10 {
        return 0.0;
    }

    let n = points.len() as f32;
    let mean = points.iter().sum::<Vector3<f32>>() / n;
    let mut m = Matrix3x3::zeros();  // 分散共分散行列
    for p in points {
        let d = p - mean;
        m += d * d.transpose();
    }
    let lambda_min = m.symmetric_eigenvalues().min();

    // 等方性（全球に一様分布した状態で1となる）
    let isotropy = 3.0 * lambda_min / m.trace();

    // 充填率
    let fill = n / BUFFER_SIZE as f32;

    isotropy * fill
}

/// 較正パラメータ計算の事後評価に用いる信頼度
/// 
/// ここでは殻らしさを評価する。
/// 点が推定中心から等距離にあるほど1に近づき、塊状の点群では0になる。
pub fn post_score(points: &[Vector3<f32>], center: Vector3<f32>) -> f32 {
    let mut sum_r2 = 0.0;
    let mut sum_r4 = 0.0;
    for p in points {
        // 補正行列まで掛けてしまうと塊上の場合であっても無理やり
        // 合わせ込めてしまうので、あえてオフセットだけの補正で評価する。
        let r2 = (p - center).norm_squared();
        sum_r2 += r2;
        sum_r4 += r2 * r2;
    }

    // k は中心から等距離なら1、等方ガウス分布の母集団では5/3。
    let k = points.len() as f32 * sum_r4 / (sum_r2 * sum_r2);
    ((5.0 / 3.0 - k) * 1.5).clamp(0.0, 1.0)
}
