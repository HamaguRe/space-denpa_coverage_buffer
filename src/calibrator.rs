//! 地磁気較正アルゴリズム
//! 
//! 楕円体フィッティングによるオフセット・スケーリング較正

use super::na::{Cholesky, SMatrix, SVector};
use super::{Vector3, Vector10, Matrix3x3};


/// 最小二乗法で点群に楕円体を当てはめ、推定中心・補正行列を返す。
///
/// 二次式 Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0 に
/// 拘束 A + B + C = 1 を入れ、C = 1 − A − B を代入すると
/// uᵀv = −z²（u = [x²−z², y²−z², 2xy, 2xz, 2yz, 2x, 2y, 2z, 1], v = [A, B, D, E, F, G, H, I, J]）
/// の線形最小二乗になる。正規方程式 (Σuuᵀ)v = −Σu z² を Cholesky 分解で解く。
///
/// トレースは平行移動で変わらないので、f32 の桁落ちを避けるため重心を引いてから解き、中心に重心を足し戻す。
/// 点が足りない・平面に潰れているなどで解けない場合や、楕円体として解釈できない場合は `None`。
pub fn fit(points: &[Vector3<f32>]) -> Option<(Vector3<f32>, Matrix3x3<f32>)> {
    let g = points.iter().sum::<Vector3<f32>>() / points.len() as f32;

    let mut m = SMatrix::<f32, 9, 9>::zeros();
    let mut rhs = SVector::<f32, 9>::zeros();
    for p in points {
        let d = p - g;
        let (x, y, z) = (d[0], d[1], d[2]);
        let u = SVector::<f32, 9>::from([
            x * x - z * z, y * y - z * z,
            2.0 * x * y, 2.0 * x * z, 2.0 * y * z,
            2.0 * x, 2.0 * y, 2.0 * z,
            1.0,
        ]);
        m += u * u.transpose();
        rhs -= u * (z * z);
    }
    let v = Cholesky::new(m)?.solve(&rhs);

    let theta = Vector10::<f32>::from([
        v[0], v[1], 1.0 - v[0] - v[1],
        v[2], v[3], v[4],
        v[5], v[6], v[7],
        v[8],
    ]);
    let (center, correction) = extract_full_ellipsoid_params(theta)?;
    Some((center + g, correction))
}

/// 推定中心 `center` と補正行列 `correction` で計測値を較正する。
pub fn calibrate(point: Vector3<f32>, center: Vector3<f32>, correction: Matrix3x3<f32>) -> Vector3<f32> {
    correction * (point - center)
}

/// θ = [A, B, C, D, E, F, G, H, I, J] から楕円体の中心と補正行列を抽出する。
///
/// 二次形式行列: Q = [[A, D, E], [D, B, F], [E, F, C]]
/// 線形項: p = [G, H, I]ᵀ
/// 定数項: J
///
/// 中心: c = −Q⁻¹p
/// 正規化: (x−c)ᵀ (Q/d) (x−c) = 1  where d = pᵀQ⁻¹p − J
/// 補正行列 W: WᵀW = Q/d  →  Cholesky(Q/d) = LLᵀ より W = Lᵀ
fn extract_full_ellipsoid_params(mut theta: Vector10<f32>) -> Option<(Vector3<f32>, Matrix3x3<f32>)> {
    if theta[0] < 0.0 { theta = -theta; }

    let (a, b, c) = (theta[0], theta[1], theta[2]);
    let (d, e, f) = (theta[3], theta[4], theta[5]);
    let (g, h, i) = (theta[6], theta[7], theta[8]);
    let j = theta[9];

    // 二次形式行列 Q と線形項 p
    let q = Matrix3x3::<f32>::new(
        a, d, e,
        d, b, f,
        e, f, c,
    );
    let p = Vector3::<f32>::new(g, h, i);

    // Q が正定値でなければ楕円体ではない
    let q_inv = q.try_inverse()?;

    // 中心: c = −Q⁻¹p
    let center = -(q_inv * p);

    // 正規化定数: d = pᵀQ⁻¹p − J > 0
    let denom = p.dot(&(q_inv * p)) - j;
    if denom <= 0.0 { return None; }

    // WᵀW = Q/d → Cholesky 分解で W = Lᵀ
    let q_normalized = q / denom;
    let chol = Cholesky::new(q_normalized)?;
    let correction = chol.l().transpose();

    Some((center, correction))
}
