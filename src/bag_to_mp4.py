"""
bag_to_mp4.py
-------------
RealSense の .bag を再生し、PNG を一切残さず

  • カラー映像              → video/color.mp4
  • 疑似カラー化した深度映像 → video/depth.mp4

の 2 本を出力するスクリプト。
----------------------------------------------
使い方例:
> python bag_to_mp4.py --bag bag/20250627_112806.bag --fps 30
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import sys

# ────────────── CLI ──────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--bag", required=True, help=".bag ファイルへのパス (例: bag/mydata.bag)"
)
parser.add_argument("--fps", type=int, default=30, help="出力 MP4 のフレームレート")
parser.add_argument("--outdir", default="video", help="MP4 の保存先フォルダ名")
args = parser.parse_args()

bag_path = args.bag
outdir = args.outdir
fps = args.fps
os.makedirs(outdir, exist_ok=True)

# ────────────── RealSense 再生初期化 ──────────────
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_path, repeat_playback=False)
profile = pipeline.start(cfg)
colorizer = rs.colorizer()  # 深度フレームを疑似カラー化

# 最初のフレームでサイズ・フォーマット確認
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()
if not color_frame or not depth_frame:
    sys.exit("カラーまたは深度ストリームが見つかりません。")

# カラーフォーマット判定（rgb8 の場合だけ後で RGB→BGR 変換）
need_RGB_to_BGR = color_frame.get_profile().format() == rs.format.rgb8

cw, ch = color_frame.get_width(), color_frame.get_height()
dw, dh = (
    colorizer.colorize(depth_frame).get_width(),
    colorizer.colorize(depth_frame).get_height(),
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # OS を問わず利用可
color_writer = cv2.VideoWriter(f"{outdir}/color_1.mp4", fourcc, fps, (cw, ch))
depth_writer = cv2.VideoWriter(f"{outdir}/depth_1.mp4", fourcc, fps, (dw, dh))

print("▶ 変換開始")
frame_idx = 0
try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        if not frames:  # 再生終了
            break

        cfrm = frames.get_color_frame()
        dfrm = frames.get_depth_frame()
        if not cfrm or not dfrm:
            continue

        # ----- カラーフレーム -----
        color = np.asanyarray(cfrm.get_data())
        if need_RGB_to_BGR:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color_writer.write(color)

        # ----- 深度フレーム（疑似カラー化） -----
        depth_vis = np.asanyarray(colorizer.colorize(dfrm).get_data())
        if depth_vis.shape[2] == 4:  # RGBA のことがある
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
        depth_writer.write(depth_vis)

        frame_idx += 1

except Exception as e:
    print("⚠️  中断:", e)

finally:
    pipeline.stop()
    color_writer.release()
    depth_writer.release()
    print(f"✅ {frame_idx} フレームを書き込み、{outdir}/color.mp4 と {outdir}/depth.mp4 を生成しました。")
