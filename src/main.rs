//! バッファを活用した地磁気センサのオンライン較正アルゴリズム

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use nalgebra::{SVector, SMatrix};
use nalgebra as na;

type Vector3<T>      = SVector<T, 3>;
type Vector10<T>     = SVector<T, 10>;
type Matrix3x3<T>    = SMatrix<T, 3, 3>;

mod buffer;
mod calibrator;
mod confidence;

use buffer::{CoverageBuffer, BUFFER_SIZE};

/// 実験データの CSV ファイル
const DATA_PATH: &str = "./magnetic_data/data-set_2.csv";

/// 事前評価：等方性 × 充足度がこの値以上なら楕円体フィッティングを行う。
const PRE_THRESHOLD: f32 = 0.2;
/// 事後評価：フィットした楕円体の中心を基準にした殻らしさがこの値以上なら、その結果を採用する。
const POST_THRESHOLD: f32 = 0.8;

/// スナップショットを出力するステップ
//const SNAPSHOT_STEPS: &[usize] = &[300, 900, 1500, 2000];  // data-set1用
//const SNAPSHOT_STEPS: &[usize] = &[1500, 2500, 3500, 4500, 5500];  // data-set1用　2回繰り返したとき
const SNAPSHOT_STEPS: &[usize] = &[1000, 3000, 6000, 9000];  // data-set2用

/// 2回目のデータに加える固定オフセット
//const OFFSET_2ND: Vector3<f32> = Vector3::new(0.2, 0.3, -0.01);

/// result_*.csv のヘッダ
const RESULT_HEADER: &str = concat!(
    "TimeStep,b_x,b_y,b_z,D11,D22,D33,D12,D13,D23,calib_x,calib_y,calib_z,",
    "pre_score,post_score,accepted,pre_threshold,post_threshold,age_min,age_median,age_mean,age_max,buffer_count\n",
);

fn main() {
    let mag_samples = load_csv(DATA_PATH);
    // 同じデータを2回繰り返して途中でオフセット値が変わった場合に適応できるか確認する。
    //let mag_samples: Vec<Vector3<f32>> = mag_samples.iter().copied()
    //    .chain(mag_samples.iter().map(|&s| s + OFFSET_2ND))
    //    .collect();

    let mut buf = CoverageBuffer::new();
    // 採用中の較正パラメータ（中心, 補正行列）
    let mut center = Vector3::zeros();
    let mut correction = Matrix3x3::identity();

    // result フォルダを一度削除して作り直す（前回のCSVが残らないようにする）
    let _ = fs::remove_dir_all("./result");
    fs::create_dir("./result").unwrap();

    // 比較描画用に元の CSV をコピーする。
    fs::copy(DATA_PATH, "./result/result_raw.csv").unwrap();

    // data_plot.py が読み込む CSV ファイルを開く
    let mut result = BufWriter::new(File::create("./result/result_full.csv").unwrap());
    result.write_all(RESULT_HEADER.as_bytes()).unwrap();

    // バッファ内エントリのage（全点・ロング形式）
    let mut age_entries = BufWriter::new(File::create("./result/result_age_entries.csv").unwrap());
    age_entries.write_all(b"TimeStep,age\n").unwrap();

    for k in 0..mag_samples.len() {
        let raw = mag_samples[k];

        // バッファ更新
        buf.update(raw);

        // バッファの中身（点とage）を集め、エントリのageを全点書き出す（グラフ描画用）
        let mut buf_points = [Vector3::zeros(); BUFFER_SIZE];
        let mut buf_ages = [0u16; BUFFER_SIZE];
        let mut n = 0;
        for entry in buf.entries() {
            buf_points[n] = entry.point;
            buf_ages[n] = entry.age;
            n += 1;
            writeln!(age_entries, "{},{}", k, entry.age).unwrap();
        }
        let points = &buf_points[..n];
        let ages = &mut buf_ages[..n];

        // 二段階の信頼度で較正パラメータを更新する。
        // 事前評価（等方性 × 充填度）を通ったら球面フィッティングの計算を行い、
        // 事後評価（推定中心基準の殻らしさ）を通ったら採用する。
        let pre_score = confidence::pre_score(points);
        let mut post_score = 0.0;
        let mut accepted = false;
        if pre_score >= PRE_THRESHOLD {
            if let Some((c, w)) = calibrator::fit(points) {
                post_score = confidence::post_score(points, c);
                if post_score >= POST_THRESHOLD {
                    (center, correction) = (c, w);
                    accepted = true;
                }
            }
        }
        let calibrated = calibrator::calibrate(raw, center, correction);

        // バッファ内のage統計（min/median/mean/max）を計算
        ages.sort_unstable();
        let age_min = ages[0];
        let age_max = ages[n - 1];
        let age_median = ages[n / 2];
        let age_mean = ages.iter().map(|&a| a as f32).sum::<f32>() / n as f32;

        // 毎ステップ result_*.csv に書き出す
        writeln!(
            result,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            k,
            center[0], center[1], center[2],
            correction[(0,0)], correction[(1,1)], correction[(2,2)],
            correction[(0,1)], correction[(0,2)], correction[(1,2)],
            calibrated[0], calibrated[1], calibrated[2],
            pre_score, post_score, u8::from(accepted),
            PRE_THRESHOLD, POST_THRESHOLD,
            age_min, age_median, age_mean, age_max,
            n,
        )
        .unwrap();

        // スナップショット出力
        if SNAPSHOT_STEPS.contains(&k) {
            println!("--- k {} ---", k);
            println!("  buffer: {}/{} points", n, BUFFER_SIZE);

            // バッファ内の点を出力
            write_buffer_snapshot(k, points, &center);

            println!("  center: [{:.4}, {:.4}, {:.4}]", center[0], center[1], center[2]);
            println!("  correction diag: [{:.4}, {:.4}, {:.4}]",
                correction[(0,0)], correction[(1,1)], correction[(2,2)]);
            println!("  pre_score: {:.4}, post_score: {:.4}", pre_score, post_score);
        }
    }
}

/// 指定ステップにおけるバッファ内の点とその時の中心座標をCSVに書き出す．
fn write_buffer_snapshot(k: usize, points: &[Vector3<f32>], center: &Vector3<f32>) {
    let path = format!("./result/result_buffer_step{}.csv", k);
    let mut f = BufWriter::new(File::create(&path).unwrap());
    f.write_all(b"x,y,z,cx,cy,cz\n").unwrap();
    for p in points {
        writeln!(f, "{},{},{},{},{},{}", p[0], p[1], p[2], center[0], center[1], center[2]).unwrap();
    }
}

/// 実験データを読み込んで，各サンプル時刻における計測値をまとめたベクトルとして返す．
fn load_csv(path: &str) -> Vec<Vector3<f32>> {
    let txt = fs::read_to_string(path).unwrap();

    let mut samples = Vec::<Vector3<f32>>::new();
    for line in txt.lines().skip(1) {  // ヘッダを読み飛ばす
        let mut tmp = Vector3::<f32>::zeros();
        for (i, row) in line.split(',').enumerate() {
            tmp[i] = row.parse().unwrap();
        }
        samples.push(tmp);
    }

    samples
}
