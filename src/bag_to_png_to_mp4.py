"""
bag_to_png_to_mp4.py
--------------------
.bag を開いて

  • カラー PNG（BGR）              frame_xxxxxx_color.png
  • 深度 PNG  （16 bit）           frame_xxxxxx_depth.png
  • 疑似カラー深度 PNG（BGR）     frame_xxxxxx_visdepth.png
  • カラー MP4   video/color.mp4
  • 疑似カラー深度 MP4 video/depth.mp4

を同時生成するスクリプト。

▼ 使い方例
python scripts/bag_to_png_to_mp4.py --bag bag/20250627_112806.bag --fps 30
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import sys

# ─────────── CLI ───────────
ap = argparse.ArgumentParser()
ap.add_argument("--bag", required=True, help=".bag ファイルへのパス")
ap.add_argument("--fps", type=int, default=30, help="出力 MP4 のフレームレート")
ap.add_argument("--framesdir", default="frames", help="PNG 保存先フォルダ")
ap.add_argument("--outdir",   default="video",  help="MP4 保存先フォルダ")
args = ap.parse_args()

bag_path   = args.bag
frames_dir = args.framesdir
video_dir  = args.outdir
fps        = args.fps

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(video_dir,  exist_ok=True)

# ─────────── RealSense 初期化 ───────────
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_path, repeat_playback=False)
profile = pipeline.start(cfg)
colorizer = rs.colorizer()

# 最初のフレームでサイズ・フォーマット確認
init_frames  = pipeline.wait_for_frames()
color_frame  = init_frames.get_color_frame()
depth_frame  = init_frames.get_depth_frame()
if not color_frame or not depth_frame:
    sys.exit("カラーまたは深度ストリームが見つかりません。")

need_swap = color_frame.get_profile().format() == rs.format.rgb8  # RGB→BGR 変換要否

cw, ch = color_frame.get_width(), color_frame.get_height()
dw, dh = colorizer.colorize(depth_frame).get_width(), colorizer.colorize(depth_frame).get_height()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
col_writer = cv2.VideoWriter(f"{video_dir}/color.mp4", fourcc, fps, (cw, ch))
dep_writer = cv2.VideoWriter(f"{video_dir}/depth.mp4", fourcc, fps, (dw, dh))

# ─────────── メインループ ───────────
print("▶ 変換開始...")
frame_idx = 0
try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        if not frames:
            break

        cfrm = frames.get_color_frame()
        dfrm = frames.get_depth_frame()
        if not cfrm or not dfrm:
            continue

        # --- カラー ---
        color = np.asanyarray(cfrm.get_data())
        if need_swap:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        else:
            color_bgr = color

        # --- 深度 & 擬似カラー ---
        depth16   = np.asanyarray(dfrm.get_data())               # 16 bit
        depth_vis = np.asanyarray(colorizer.colorize(dfrm).get_data())
        if depth_vis.shape[2] == 4:          # RGBA→BGR
            depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGBA2BGR)
        else:                                # RGB →BGR
            depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

        # ---------- PNG ----------
        stem = f"{frames_dir}/20250704_DepthErrorExpt/frame_1m_{frame_idx:06d}"
        cv2.imwrite(f"{stem}_color.png",    color_bgr)
        cv2.imwrite(f"{stem}_depth.png",    depth16)
        cv2.imwrite(f"{stem}_visdepth.png", depth_bgr)

        # ---------- MP4 ----------
        col_writer.write(color_bgr)
        dep_writer.write(depth_bgr)

        frame_idx += 1

except Exception as e:
    print("⚠️  中断:", e)

finally:
    pipeline.stop()
    col_writer.release()
    dep_writer.release()
    print(f"✅ {frame_idx} フレームを書き出し完了（PNG & MP4）。")
