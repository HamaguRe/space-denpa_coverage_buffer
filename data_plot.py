# result_*.csvを読み込んでグラフ作成

import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# CSVファイルを読んでヘッダと値を対応させた辞書型を返す（値はnumpy配列）
#
# --- CSV File ---
# TimeStep,x,y
# 0,0.01,-0.1
# 1,0.02,-0.2
# 2,0.03,-0.3
# ---- Return -----
# {'TimeStep': array([0,1,2]), 'x': array([0.01,0.02,0.03]), 'y': array([-0.1,-0.2,-0.3])}
def load_csv(file_path):
    data_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                data_dict.setdefault(key, []).append(float(value))

    return {k: np.array(v) for k, v in data_dict.items()}


# 同じx軸に対して複数の系列をまとめて描画するヘルパー
def plot_series(ax, x, data, keys, labels=None, loc="lower right"):
    labels = labels if labels is not None else keys
    for key, lb in zip(keys, labels):
        ax.plot(x, data[key], label=lb)
    ax.set_xlim(0, x[-1])
    ax.legend(loc=loc)


# 3D軸の表示範囲・ラベル・凡例をまとめて設定するヘルパー
def setup_3d_axes(ax, lim, legend_loc="upper right"):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend(loc=legend_loc)


# result_buffer_step{N}.csv のファイル名から N を抽出する
def step_from_path(p):
    return int(os.path.basename(p).replace("result_buffer_step", "").replace(".csv", ""))


# 較正前のデータを読み込む（比較表示用）
mag_raw = load_csv("./magnetic_data/data-set_2.csv")

# 較正結果のデータを読み込む
result = load_csv("./result/result_full.csv")

# スケールファクタとバイアスの時間遷移を表示するウィンドウ
fig1 = plt.figure(figsize = (14, 10))
fig1.suptitle("Time evolution of estimated parameters", fontsize=16)

ax_bias       = fig1.add_subplot(2, 2, 1, xlabel="Time step", ylabel="Bias [Gauss]")
ax_scale      = fig1.add_subplot(2, 2, 2, xlabel="Time step", ylabel="Scale factor")
ax_confidence = fig1.add_subplot(2, 2, 3, xlabel="Time step", ylabel="Confidence score")
ax_age        = fig1.add_subplot(2, 2, 4, xlabel="Time step", ylabel="Age [ticks]")
ax_bias.set_title("Bias")
ax_scale.set_title("Scale factor")
ax_confidence.set_title("Confidence")
ax_age.set_title("Buffer entry age")

# --- 描画 ---
k = result["TimeStep"]  # 計算ステップ

# バイアス・スケールファクタ
plot_series(ax_bias,  k, result, ["b_x", "b_y", "b_z"], labels=["x", "y", "z"])
plot_series(ax_scale, k, result, ["D11", "D22", "D33"])
for key in ["D12", "D13", "D23"]:
    ax_scale.plot(k, result[key], linestyle="--", label=key)
ax_scale.legend(loc="lower right")

# 信頼度スコア
ax_confidence.plot(k, result["confidence"], color="purple", label="confidence")
ax_confidence.set_xlim(0, k[-1])
ax_confidence.set_ylim(0.0, 1.0)
ax_confidence.legend(loc="lower right")

# バッファ内エントリのage推移
age_entries = load_csv("./result/result_age_entries.csv")

# 個別エントリのage（散布図・半透明）を先に描画（_age_stepごとに間引いて表示）
_age_step = 10
_ts_all   = age_entries["TimeStep"]
_ages_all = age_entries["age"]
_mask     = (_ts_all.astype(int) % _age_step) == 0
ax_age.scatter(_ts_all[_mask], _ages_all[_mask], s=1, color="green", alpha=0.3, zorder=1)

# min/median/maxの線を散布図の上に重ねる
ax_age.plot(k, result["age_max"],    color="tomato",     label="max",    zorder=2)
ax_age.plot(k, result["age_mean"],   color="purple",   label="mean",   zorder=2)
ax_age.plot(k, result["age_median"], color="darkorange", label="median", zorder=2)
ax_age.plot(k, result["age_min"],    color="steelblue",  label="min",    zorder=2)
ax_age.set_xlim(0, k[-1])
ax_age.set_ylim(bottom=0)

# 右軸: バッファ内の点の個数
ax_count = ax_age.twinx()
ax_count.set_ylabel("Buffer count [points]")
h_count, = ax_count.plot(k, result["buffer_count"], color="gray", linestyle="--", label="Buffer count", zorder=3)
ax_count.set_ylim(0, 17)  # BUFFER_SIZE=16 + 余白1

# 凡例を左軸側にまとめる
handles_left, labels_left = ax_age.get_legend_handles_labels()
ax_age.legend(handles_left + [h_count], labels_left + ["Buffer count"], loc="upper left")

# --- 元データと較正したデータを3Dプロットで比較するウィンドウ
fig2 = plt.figure(figsize = (10, 5))

ax_raw  = fig2.add_subplot(1, 2, 1, projection="3d")
ax_cal  = fig2.add_subplot(1, 2, 2, projection="3d")
ax_raw.set_title("Raw data")
ax_cal.set_title("Calibrated data")

# 座標系の原点
ax_raw.scatter([0],[0],[0], marker="o", color="r", label="Origin")
ax_cal.scatter([0],[0],[0], marker="o", color="r", label="Origin")

ax_raw.scatter(mag_raw["x"], mag_raw["y"], mag_raw["z"], s=5, label="Raw data")
ax_cal.scatter(result["calib_x"], result["calib_y"], result["calib_z"], s=5, color="darkorange", label="Calibrated data")

lim = 1.0
for ax in [ax_raw, ax_cal]:
    setup_3d_axes(ax, lim)

# --- バッファ内の計測点をスナップショットごとに3Dプロットするウィンドウ
buffer_files = sorted(
    glob.glob("./result/result_buffer_step*.csv"),
    key=step_from_path,
)

if buffer_files:
    fig3 = plt.figure(figsize=(5 * len(buffer_files), 5))
    fig3.suptitle("Buffer contents at each snapshot", fontsize=16)

    # 単位球メッシュは全スナップショットで共通なのでループ外で1回だけ作る
    _u = np.radians(np.arange(0, 361, 15))
    _v = np.radians(np.arange(0, 181, 15))
    _uu, _vv = np.meshgrid(_u, _v)
    sphere_x = np.sin(_vv) * np.cos(_uu)
    sphere_y = np.sin(_vv) * np.sin(_uu)
    sphere_z = np.cos(_vv)

    for i, fpath in enumerate(buffer_files):
        step = step_from_path(fpath)
        buf_data = load_csv(fpath)

        # centerで正規化（中心化→単位ベクトル化）
        center = np.array([buf_data["cx"][0], buf_data["cy"][0], buf_data["cz"][0]])
        pts = np.column_stack([buf_data["x"], buf_data["y"], buf_data["z"]]) - center
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        norm_x, norm_y, norm_z = pts[:, 0], pts[:, 1], pts[:, 2]

        ax = fig3.add_subplot(1, len(buffer_files), i + 1, projection="3d")
        ax.set_title(f"Step {step}")

        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="lightgray", linewidth=0.5, alpha=0.5)

        ax.scatter([0], [0], [0], marker="o", color="r", label="Origin")
        ax.scatter(norm_x, norm_y, norm_z, s=20, label="Buffer points")
        setup_3d_axes(ax, lim)

plt.show()
