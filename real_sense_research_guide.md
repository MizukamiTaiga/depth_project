# RealSense D457 研究スターターガイド *(attach this .md at the start of every ChatGPT session)*

> 最終更新 2025‑07‑04 — **本ファイルをそのまま添付すると、以後のチャットで前提が共有されます。**

---

## 0  背景と目的

本研究では カラー画像, Depth画像, 音響情報からランドマークを検出し, 視覚障害者のナビゲーションシステムを構築することを目的としている. 現在は, まず, カラー画像, Depth画像からランドマークを検出するために研究をしている. Intel RealSense D457 を用いて「カラー (1280×800) と深度 (1280×720) を **完全に同一ピクセル座標** で解析」する。\
データ取得・前処理・可視化・深度取得までを **再現性 100 %** で回せるよう、ディレクトリ構成・スクリプト・命名規約を固定する。

---

## 1  プロジェクト構成

```
depth_project/
├─ bag/                # 録画 *.bag (ground‑truth)
├─ scripts/            # 公式ユーティリティ (§4)
├─ captures/           # PNG / log / csv 一時出力
├─ venv310/            # Python 3.10 venv (ラップトップPCで使用する仮想環境, 標準使用)
├─ venv_desktop/       # Python 3.10 venv (DesktopPCで使用する仮想環境)
└─ requirements.txt    # 固定バージョン (§2)
```

---

## 2  Python 環境 & 依存パッケージ

```ps1
python -3.10 -m venv venv310      # 3.10 固定
./venv310/Scripts/activate        # Linux: source venv310/bin/activate
pip install -r requirements.txt
```

`requirements.txt` （GUI あり版 OpenCV で固定）：

```
numpy==2.2.6
opencv-python==4.11.0.86      # headless 版は入れない
pyrealsense2==2.55.1.6486
setuptools==65.5.0

```

> *OpenCV‑GUI が不要なサーバでは headless 版を明示可。ただし GUI スクリプトは使えない。*

---

## 3  座標統一ルール — \`\`\*\* 詳細ガイド\*\*

Intel RealSense のストリームは各センサーが独立した解像度と光軸を持つため、**生のままではピクセル座標が一致しない**。本研究では常に `rs.align` を使い、**深度をカラー解像度 (1280 × 800) へリプロジェクション**して解析・学習を行う。

### 3.1  どのストリームに合わせるか

| ケース             | コード                                 | 出力解像度    | 理由                                                 |
| --------------- | ----------------------------------- | -------- | -------------------------------------------------- |
| **標準 (本研究すべて)** | `align = rs.align(rs.stream.color)` | 1280×800 | カラー主観でセマンティクスを扱うため／OpenCV 等の標準 16:10 フレームワークと親和性高い |
| 動作確認・高速プレビュー    | `align = rs.align(rs.stream.depth)` | 1280×720 | UI 表示を軽くしたいとき。**研究成果には用いない**                       |

> `align.process(frames)` を呼ぶと **下記 4 手順** が自動で実行される：
>
> 1. 深度 → 点群 (XYZ, 左カメラ座標系)
> 2. 点群をカラーセンサー座標系へ外部変換 (Extrinsics 検出済)
> 3. カラーの内部パラメータで 2D 投影 (1280×800)
> 4. Z (mm) を最短距離の対応ピクセルへ書き込み

### 3.2  コード雛形

```python
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipe.start(cfg)

align = rs.align(rs.stream.color)  # ← 座標系統一
while True:
    frames = align.process(pipe.wait_for_frames())
    color  = frames.get_color_frame()          # 1280×800
    depth  = frames.get_depth_frame()          # 1280×800, invalid=0
    # --- 解析処理へ ---
```

### 3.3  無効ピクセル (invalid=0) の扱い

- 外周の 80 px 以内に**必ず 0 深度帯が出る** → `depth.get_distance(x,y)==0` で判定しマスク
- 例：OpenCV マスク生成
  ```python
  depth_np = np.asanyarray(depth.get_data())
  mask = depth_np == 0          # True = invalid
  valid_depth = np.ma.masked_array(depth_np, mask)  # NumPy マスク配列
  ```

### 3.4  3D ⇄ 2D 変換ユーティリティ

```python
intr = color.profile.as_video_stream_profile().get_intrinsics()
# 2D→3D
pt3 = rs.rs2_deproject_pixel_to_point(intr, [px, py], depth_mm/1000)
# 3D→2D
px, py = rs.rs2_project_point_to_pixel(intr, pt3)
```

深度(mm) → m に換算して渡す点に注意。

### 3.5  パフォーマンス

| 解像度              | align コスト (CPU)            | 備考           |
| ---------------- | -------------------------- | ------------ |
| 1280×800\@30 fps | 4–8 ms / frame (i7‑12700H) | 約 10–15 % 余力 |
| 848×480\@30 fps  | <3 ms                      | 軽負荷検証用       |

> 高速化が必要なら
>
> 1. ROI 切り出し
> 2. f‑z カリング (遠距離はマスク)
> 3. `rs.decimation_filter` で深度ダウンサンプル

### 3.6  よくある落とし穴

1. **Viewer を開いたまま align コードを動かすとフレームが来ない** → Viewer を閉じる。
2. `max_distance` を狭めると遠距離の Z が 0 扱いになる → 可視カラーマップだけの影響、raw Z は保持される。
3. align 後の **カラーと深度の Intrinsics は同一** だが、DistortionCoeffs はカラー準拠になる点に注意。

---

## 4  標準スクリプト（`scripts/`）  標準スクリプト（`scripts/`）

| ファイル                           | 主な引数                                  | 概要                                                                                             |
| ------------------------------ | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **capture\_5frame.py**         | `python capture_5frame.py <basename>` | D457 から **5 フレームだけ** を録画し `bag/<basename>.bag` に保存 (raw ストリーム, `rs.align` 不要)                  |
| **export\_png.py**             | `--bag`, `--outdir`                   | 指定 `.bag` を \`\`\*\* で整列\*\*して展開。カラー PNG / 深度 16‑bit PNG / 疑似カラー PNG を連番出力し、機械学習データセット化を容易にする。 |
| **depth\_view\_click\_bag.py** | `--bag`                               | `.bag` を再生し、整列深度を GUI 表示＋クリック位置の **mm 値** を表示。検証時に使用。                                          |

> **新規テンプレート追加ルール**
>
> 1. 英小文字 + 下線スネーク命名。
> 2. 必ず `argparse` CLI 化。
> 3. `rs.align` 使用時はコメントで “座標系保証” を明示。

---

## 5  capture\_5frame.py — 概要

フルコードは `scripts/capture_5frame.py` に保存。 ポイントのみ：

```python
cfg.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_record_to_file(f"bag/{basename}.bag")
# 5 フレームだけ loop
```

実験用に **他スクリプトも同ディレクトリに配置**し、必要に応じて改変していくこと。

\---  capture\_5frame.py — 全コード

```python
"""Record exactly 5 aligned frames into <basename>.bag"""
import argparse, os, pyrealsense2 as rs

parser = argparse.ArgumentParser()
parser.add_argument("basename", help="bag/<basename>.bag を生成")
args = parser.parse_args()

os.makedirs("bag", exist_ok=True)
bag_path = f"bag/{args.basename}.bag"

cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# 録画ファイル設定
cfg.enable_record_to_file(bag_path)

pipe = rs.pipeline(); pipe.start(cfg)
# --- 5 frames ---
for i in range(5):
    pipe.wait_for_frames()
pipe.stop()
print("saved", bag_path)
```

---

## 6  命名規約

```
{YYYYMMDD}_{session}_{index}.bag         # bag
frame_{n:06d}_{color|depth|vis}.png      # PNG
{basename}_clicklog.csv                 # クリックログ
```

---

## 7  研究チェックリスト 

・

---

