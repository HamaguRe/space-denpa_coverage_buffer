//! バッファを活用した地磁気センサのオンライン較正アルゴリズム

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use nalgebra::{SVector, SMatrix};
use nalgebra as na;

type Vector3<T>      = SVector<T, 3>;
type Vector10<T>     = SVector<T, 10>;
type Matrix3x3<T>    = SMatrix<T, 3, 3>;
type Matrix10x10<T>  = SMatrix<T, 10, 10>;

mod calibrator;
mod buffer;

use buffer::{CoverageBuffer, BUFFER_SIZE};
use calibrator::LeastSquaresFull;

/// 信頼度がこの値より大きければ楕円フィッティング計算をする
const CONFIDENCE_THR: f32 = 0.1;

/// スナップショットを出力するステップ
//const SNAPSHOT_STEPS: &[usize] = &[300, 900, 1500, 2000];  // data-set1用
const SNAPSHOT_STEPS: &[usize] = &[1000, 3000, 6000, 9000];  // data-set2用

/// 2回目のデータに加える固定オフセット
const OFFSET_2ND: Vector3<f32> = Vector3::new(0.2, 0.3, -0.01);

/// result_*.csv のヘッダ
const RESULT_HEADER: &[u8] = b"TimeStep,b_x,b_y,b_z,D11,D22,D33,D12,D13,D23,calib_x,calib_y,calib_z,confidence,age_min,age_median,age_mean,age_max,buffer_count\n";

fn main() {
    let mag_samples = load_csv("./magnetic_data/data-set_2.csv");
    // 同じデータを2回繰り返して途中でオフセット値が変わった場合に適応できるか確認する。
    //let mag_samples: Vec<Vector3<f32>> = mag_samples.iter().copied()
    //    .chain(mag_samples.iter().map(|&s| s + OFFSET_2ND))
    //    .collect();

    let mut buf = CoverageBuffer::new();
    let mut cal = LeastSquaresFull::new();

    // result フォルダを一度削除して作り直す（前回のCSVが残らないようにする）
    let _ = fs::remove_dir_all("./result");
    fs::create_dir("./result").unwrap();

    // data_plot.py が読み込む CSV ファイルを開く
    let mut result = BufWriter::new(File::create("./result/result_full.csv").unwrap());
    result.write_all(RESULT_HEADER).unwrap();

    // バッファ内エントリのage（全点・ロング形式）
    let mut age_entries = BufWriter::new(File::create("./result/result_age_entries.csv").unwrap());
    age_entries.write_all(b"TimeStep,age\n").unwrap();

    for k in 0..mag_samples.len() {
        let raw = mag_samples[k];

        // バッファに投入（角度比較は現在のオフセット推定値で中心化して行う）
        let (center, _) = cal.params();
        buf.update(raw, center);

        // バッファ内エントリのageを全点書き出す
        for slot in &buf.buffer {
            if let Some(e) = slot {
                writeln!(age_entries, "{},{}", k, e.age).unwrap();
            }
        }

        // バッファの中身を集めて較正器を更新
        let buf_points = buf.points();
        let n = buf.count();
        let confidence_score = buf.confidence_score;
        if n == BUFFER_SIZE && confidence_score > CONFIDENCE_THR {
            cal.update(&buf_points[..n]);
        }

        // 較正パラメータと較正値を取得
        let (center, correction) = cal.params();
        let calibrated = cal.calibrate(raw);
        let confidence = buf.confidence_score;

        // バッファ内のage統計（min/median/mean/max）を計算
        let (age_min, age_median, age_mean, age_max) = {
            let mut ages: Vec<u16> = buf.buffer.iter()
                .filter_map(|s| s.map(|e| e.age))
                .collect();
            if ages.is_empty() {
                (0u16, 0u16, 0.0f32, 0u16)
            } else {
                ages.sort_unstable();
                let min = *ages.first().unwrap();
                let max = *ages.last().unwrap();
                let median = ages[ages.len() / 2];
                let mean = ages.iter().map(|&a| a as f32).sum::<f32>() / ages.len() as f32;

                (min, median, mean, max)
            }
        };

        // 毎ステップ result_*.csv に書き出す
        let row = format!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            k,
            center[0], center[1], center[2],
            correction[(0,0)], correction[(1,1)], correction[(2,2)],
            correction[(0,1)], correction[(0,2)], correction[(1,2)],
            calibrated[0], calibrated[1], calibrated[2],
            confidence,
            age_min, age_median, age_mean, age_max,
            n,
        );
        result.write_all(row.as_bytes()).unwrap();

        // スナップショット出力
        if SNAPSHOT_STEPS.contains(&k) {
            println!("--- k {} ---", k);
            println!("  buffer: {}/{} points", n, BUFFER_SIZE);

            // バッファ内の点を出力
            write_buffer_snapshot(k, &buf_points[..n], &center);

            println!("  center: [{:.4}, {:.4}, {:.4}]", center[0], center[1], center[2]);
            println!("  correction diag: [{:.4}, {:.4}, {:.4}]",
                correction[(0,0)], correction[(1,1)], correction[(2,2)]);
            println!("  confidence: {:.4}", confidence);
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
