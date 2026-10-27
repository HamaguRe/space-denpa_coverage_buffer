//! 地磁気較正アルゴリズム
//! 
//! 楕円体フィッティングによるオフセット・スケーリング較正

use super::na::{SymmetricEigen, Cholesky};
use super::{
    Vector3, Vector10,
    Matrix3x3, Matrix10x10
};


/// 最小二乗法（一般楕円体版）による較正
///
/// 交差項を含む 10 パラメータの楕円体を当てはめる。
/// 散布行列 M = (1/N)Σ ξξᵀ の最小固有値に対応する固有ベクトルを θ とする。
pub struct LeastSquaresFull {
    center: Vector3<f32>,
    correction: Matrix3x3<f32>,
}

impl LeastSquaresFull {
    pub fn new() -> Self {
        LeastSquaresFull {
            center: Vector3::<f32>::zeros(),
            correction: Matrix3x3::<f32>::identity(),
        }
    }

    pub fn update(&mut self, points: &[Vector3<f32>]) {
        let n = points.len();

        let mut xi = Vector10::<f32>::zeros();
        let mut m = Matrix10x10::<f32>::zeros();
        for i in 0..n {
            let x = points[i][0];
            let y = points[i][1];
            let z = points[i][2];

            xi[0] = x * x;
            xi[1] = y * y;
            xi[2] = z * z;
            xi[3] = 2.0 * x * y;
            xi[4] = 2.0 * x * z;
            xi[5] = 2.0 * y * z;
            xi[6] = 2.0 * x;
            xi[7] = 2.0 * y;
            xi[8] = 2.0 * z;
            xi[9] = 1.0;

            m += xi * xi.transpose();
        }
        m *= 1.0 / n as f32;

        let eigen = SymmetricEigen::new(m);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        let min_idx = eigenvalues.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let theta: Vector10<f32> = eigenvectors.column(min_idx).into();

        let Some((center, correction)) = extract_full_ellipsoid_params(&theta) else { return };

        self.center = center;
        self.correction = correction;
    }

    pub fn calibrate(&self, point: Vector3<f32>) -> Vector3<f32> {
        self.correction * (point - self.center)
    }

    /// 現在の推定パラメータ（中心, 補正行列）を返す
    pub fn params(&self) -> (Vector3<f32>, Matrix3x3<f32>) {
        (self.center, self.correction)
    }
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
fn extract_full_ellipsoid_params(theta: &Vector10<f32>) -> Option<(Vector3<f32>, Matrix3x3<f32>)> {
    let mut th = *theta;
    if th[0] < 0.0 { th = -th; }

    let (a, b, c) = (th[0], th[1], th[2]);
    let (d, e, f) = (th[3], th[4], th[5]);
    let (g, h, i) = (th[6], th[7], th[8]);
    let j = th[9];

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
