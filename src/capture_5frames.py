#!/usr/bin/env python
"""
capture_5frame.py
-----------------
RealSense から 5 フレーム分だけ取得して 1 本の .bag ファイルに保存するスクリプト。

使い方：
    python capture_5frame.py <base_name>

例：
    python scripts/capture_5frames.py test01
        → bag/test01.bag が生成される
"""
import time, os, sys, argparse, pyrealsense2 as rs

# ---------- Argument ----------
ap = argparse.ArgumentParser()
ap.add_argument("basename")
ap.add_argument("-n", "--num", type=int, default=5, help="frames to capture")
args = ap.parse_args()

out = f"bag/{args.basename}.bag"
if os.path.exists(out):
    sys.exit(f"⚠ {out} already exists")

os.makedirs("bag", exist_ok=True)

# ---------- Pipeline ----------
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_record_to_file(out)
profile = pipe.start(cfg)

try:
    # auto-exposure settle
    for _ in range(30):
        pipe.wait_for_frames()

    print(f"▶ capturing {args.num} frames …")
    saved = 0
    while saved < args.num:
        f = pipe.wait_for_frames()
        if f.get_depth_frame() and f.get_color_frame():
            saved += 1
            print(f"  ✓ {saved}/{args.num}")
finally:
    pipe.stop()
print("✅ done:", out)