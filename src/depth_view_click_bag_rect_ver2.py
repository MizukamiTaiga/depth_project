# depth_view_rect_avg.py  —  interactive viewer with rectangle depth averaging (v0.4)
# -----------------------------------------------------------------
#   1.  --bag / --start で .bag と開始フレームを指定
#   2.  右矢印キー(→) : +1 frame   ← : -1 frame  ※直前フレームはキャッシュで即時表示
#   3.  マウス左クリック×2 : 対角２点を指定し矩形領域内の平均深度 (mm) を表示
#        •  1 回目クリックで始点 (p1) ／同じフレーム内でいつでも新しい選択を開始可能
#        •  2 回目クリックで終点 (p2) → 平均深度を計算して矩形と平均値を **次の ← / → キーが押されるまで持続表示**
#        •  矩形は “カラー側” と “深度側” **両方** に重ね描き
#   4.  ESC で終了
# -----------------------------------------------------------------
# v0.4 変更点
# ● クリック確定後の矩形 / テキストを **左右両画像** へ表示
#     - カラー表示: そのまま描画
#     - 深度表示: X 座標を +DISP_W してオフセット描画
# ● HUD 表示処理を draw_hud() に分離
# -----------------------------------------------------------------
#python scripts/depth_view_click_bag_rect_ver2.py --bag bag/20250630_105842.bag --start 0

import argparse
import sys
from datetime import timedelta
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

# -------------------------------------------------
# 1. CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="RealSense .bag viewer with rectangle depth averaging")
parser.add_argument("--bag", "-b", required=True, help="Path to .bag file")
parser.add_argument("--start", "-s", type=int, default=0, help="Start frame index (default=0)")
args = parser.parse_args()

bag_path = args.bag
start_frame = max(args.start, 0)

# -------------------------------------------------
# 2. RealSense 初期化
# -------------------------------------------------

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_path, repeat_playback=False)
profile = pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# ---- FPS を取得 ------------------
try:
    video_profile = next(p for p in profile.get_streams()
                         if isinstance(p, rs.video_stream_profile))
except StopIteration:
    frm = pipeline.wait_for_frames()
    vfrm = frm.get_color_frame() or frm.get_depth_frame()
    if vfrm is None:
        print("⚠ video_stream_profile が取得できません。終了します。")
        pipeline.stop(); sys.exit(1)
    video_profile = vfrm.get_profile().as_video_stream_profile()

fps = video_profile.fps()
dt_us = 1e6 / fps
# -------------------------------------------------

# 指定フレームまでスキップ
for _ in range(start_frame):
    pipeline.wait_for_frames()

print(f"▶ Interactive viewer ready — ESC to quit (start={start_frame})\nFILE: {bag_path}")

colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 0)
colorizer.set_option(rs.option.max_distance, 16.0)

# 表示サイズ
DISP_W, DISP_H = 640, 480

# キャッシュ
history: deque[dict] = deque(maxlen=120)

# グローバルなフレーム情報
_depth_raw = None
_scale_x = _scale_y = 1.0
_w_raw = _h_raw = 0

# 矩形選択状態
pt1 = pt2 = None  # raw sensor coords
last_avg_mm: float | None = None

# 解像度表示フラグ
printed_res: bool = False

# -------------------------------------------------
# 3. Mouse callback — 矩形平均深度
# -------------------------------------------------

def on_mouse(event, x, y, flags, param):
    global pt1, pt2, last_avg_mm

    if event != cv2.EVENT_LBUTTONDOWN or _depth_raw is None:
        return

    # 表示→センサ座標
    raw_x = int(x / _scale_x) if x < DISP_W else int((x - DISP_W) / _scale_x)
    raw_y = int(y / _scale_y)

    if not (0 <= raw_x < _w_raw and 0 <= raw_y < _h_raw):
        print(f"⚠ 範囲外クリック: ({raw_x}, {raw_y})")
        pt1 = pt2 = None; last_avg_mm = None
        return

    # クリック状態遷移
    if pt1 is None or (pt1 is not None and pt2 is not None):
        pt1 = (raw_x, raw_y)
        pt2 = None
        last_avg_mm = None
        print(f"始点: {pt1}")
        return

    if pt1 is not None and pt2 is None:
        pt2 = (raw_x, raw_y)
        x1, y1 = pt1; x2, y2 = pt2
        x_lo, x_hi = sorted([x1, x2])
        y_lo, y_hi = sorted([y1, y2])

        roi = _depth_raw[y_lo:y_hi+1, x_lo:x_hi+1]
        valid = roi[roi > 0]
        if valid.size == 0:
            print("⚠ ROI 内に有効な深度がありません。")
            pt1 = pt2 = None; last_avg_mm = None
        else:
            last_avg_mm = float(valid.mean())
            print(f"【矩形平均深度】({pt1})–({pt2}) → {last_avg_mm:.1f} mm  (n={valid.size})")
        return

# -------------------------------------------------
# 4. 1 フレーム取得
# -------------------------------------------------

def fetch_frame():
    global _depth_raw, _scale_x, _scale_y, _w_raw, _h_raw, printed_res

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    _depth_raw = np.asanyarray(depth_frame.get_data())
    depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_img = np.asanyarray(color_frame.get_data())

    if color_frame.get_profile().format() == rs.format.rgb8:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    if depth_vis.shape[2] == 4:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGBA2BGR)
    else:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

    # ---- ★ 初回のみ解像度を表示 ----------
    if not printed_res:
        h_raw, w_raw = _depth_raw.shape
        h_color, w_color = color_img.shape[:2]
        print(f"Current bag resolution  ->  Color: {w_color}×{h_color}  Depth: {w_raw}×{h_raw} (after align)")
        printed_res = True
    # ----------------------------------------

    _h_raw, _w_raw = _depth_raw.shape
    _scale_x = DISP_W / _w_raw
    _scale_y = DISP_H / _h_raw

    color_disp = cv2.resize(color_img, (DISP_W, DISP_H))
    depth_disp = cv2.resize(depth_vis, (DISP_W, DISP_H))
    combined = np.hstack((color_disp, depth_disp))

    history.append(dict(
        frame_idx=frame_idx,
        combined=combined,
        depth_raw=_depth_raw.copy(),
        scale_x=_scale_x,
        scale_y=_scale_y,
        w_raw=_w_raw,
        h_raw=_h_raw,
    ))

    return combined

# -------------------------------------------------
# 5. HUD 描画
# -------------------------------------------------

def draw_hud(img: np.ndarray) -> np.ndarray:
    disp = img.copy()

    # 1 点のみ → 印
    if pt1 is not None and pt2 is None:
        x_disp = int(pt1[0] * _scale_x)
        y_disp = int(pt1[1] * _scale_y)
        for offset in (0, DISP_W):  # 両側
            cv2.circle(disp, (x_disp + offset, y_disp), 3, (0, 255, 255), -1)

    # 2 点確定 → 矩形
    if pt1 is not None and pt2 is not None:
        x1d, y1d = int(pt1[0] * _scale_x), int(pt1[1] * _scale_y)
        x2d, y2d = int(pt2[0] * _scale_x), int(pt2[1] * _scale_y)
        for offset in (0, DISP_W):
            cv2.rectangle(disp, (x1d + offset, y1d), (x2d + offset, y2d), (0, 255, 0), 1)
        if last_avg_mm is not None:
            txt = f"{last_avg_mm:.1f} mm"
            tx, ty = min(x1d, x2d) + 5, min(y1d, y2d) - 10
            cv2.putText(disp, txt, (tx, max(ty, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return disp

# -------------------------------------------------
# 6. メインループ
# -------------------------------------------------

frame_idx = start_frame

cv2.namedWindow("depth_view", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("depth_view", on_mouse)
combined = fetch_frame()
if combined is None:
    print("⚠ 最初のフレームが取得できませんでした。終了します。")
    pipeline.stop(); sys.exit(1)
cv2.resizeWindow("depth_view", combined.shape[1], combined.shape[0])

KEY_RIGHT = {2555904, 65363, 83}
KEY_LEFT  = {2424832, 65361, 81}
ESC = 27

while True:
    cv2.imshow("depth_view", draw_hud(combined))

    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    if key == ESC:
        break

    elif key in KEY_RIGHT:
        frame_idx += 1
        new_frame = fetch_frame()
        if new_frame is not None:
            combined = new_frame
            pt1 = pt2 = None; last_avg_mm = None
            print(f"Frame {frame_idx} displayed (→)")

    elif key in KEY_LEFT:
        if frame_idx == 0:
            continue
        if len(history) >= 2 and history[-2]['frame_idx'] == frame_idx - 1:
            history.pop()
            prev = history[-1]
            combined = prev['combined']
            _depth_raw = prev['depth_raw']
            _scale_x, _scale_y = prev['scale_x'], prev['scale_y']
            _w_raw, _h_raw = prev['w_raw'], prev['h_raw']
            frame_idx -= 1
            pt1 = pt2 = None; last_avg_mm = None
            print(f"Frame {frame_idx} displayed from cache (←)")
            continue

        frame_idx -= 1
        playback.pause()
        playback.seek(timedelta(microseconds=int(frame_idx * dt_us)))
        playback.resume()

        flushed = 0
        while flushed < 5 and pipeline.poll_for_frames():
            flushed += 1

        while history and history[-1]['frame_idx'] > frame_idx:
            history.pop()

        new_frame = fetch_frame()
        if new_frame is not None:
            combined = new_frame
            pt1 = pt2 = None; last_avg_mm = None
            print(f"Frame {frame_idx} displayed (←, flushed {flushed})")

cv2.destroyAllWindows()
pipeline.stop()
