# depth_view_click_bag.py  —  minimal interactive viewer (v0.5 cache)
# -----------------------------------------------------------------
# 1. --bag / --start で .bag と開始フレームを指定
# 2. 右矢印キー(→) : +1 frame   ← : -1 frame  ※直前フレームはキャッシュで即時表示
# 3. マウス左クリック : (x,y) の深度値(mm) を表示
# 4. ESC で終了
# -----------------------------------------------------------------
# 変更点 (v0.4 → v0.5)
# ● 直近 N=30 フレームをリングバッファに保持
#   ← を押した際にキャッシュにあれば再デコードせず即表示
# ● キャッシュに無い場合のみ playback.seek()＋flush を実行
#   → 大容量 .bag でも 1 フレーム戻るだけならノーウェイト
# -----------------------------------------------------------------
#python scripts\depth_view_click_bag.py --bag bag\【ファイル名】.bag --start 0

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
parser = argparse.ArgumentParser(description="RealSense .bag viewer (seek + cache)")
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

# ---- FPS を取得（フレーム側フォールバックつき） ------------------
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

fps = video_profile.fps()            # 例: 30
dt_us = 1e6 / fps
# ------------------------------------------------------------------

# 指定フレームまでスキップ
for _ in range(start_frame):
    pipeline.wait_for_frames()

print(f"▶ Interactive viewer ready — ESC to quit (start={start_frame})\nFILE: {bag_path}")

colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 0)    # custom
colorizer.set_option(rs.option.max_distance, 16.0)  # 16 m 上限

# 表示サイズ
DISP_W, DISP_H = 640, 480

# 全フレーム情報を保持するための構造体
autostore = dict  # alias

# リングバッファ（直近 N フレーム）
history: deque[dict] = deque(maxlen=120)

# グローバルでクリック用に保持（最新フレームのデータ）
_depth_raw = None
_scale_x = _scale_y = 1.0
_w_raw = _h_raw = 0

# -------------------------------------------------
# 3. Mouse callback — クリックで深度表示
# -------------------------------------------------

def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or _depth_raw is None:
        return
    raw_x = int(x / _scale_x) if x < DISP_W else int((x - DISP_W) / _scale_x)
    raw_y = int(y / _scale_y)
    if 0 <= raw_x < _w_raw and 0 <= raw_y < _h_raw:
        d = _depth_raw[raw_y, raw_x]
        area = "カラー" if x < DISP_W else "深度"
        print(f"【{area}クリック】({raw_x}, {raw_y}) → depth {d} mm")
    else:
        print(f"⚠ 範囲外クリック: ({raw_x}, {raw_y})")

# -------------------------------------------------
# 4. 1 フレーム取得し画像合成
# -------------------------------------------------

def fetch_frame():
    global _depth_raw, _scale_x, _scale_y, _w_raw, _h_raw

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    _depth_raw = np.asanyarray(depth_frame.get_data())
    depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_img = np.asanyarray(color_frame.get_data())

    # RGB→BGR
    if color_frame.get_profile().format() == rs.format.rgb8:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    if depth_vis.shape[2] == 4:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGBA2BGR)
    else:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

    # リサイズ率計算
    _h_raw, _w_raw = _depth_raw.shape
    _scale_x = DISP_W / _w_raw
    _scale_y = DISP_H / _h_raw

    color_disp = cv2.resize(color_img, (DISP_W, DISP_H))
    depth_disp = cv2.resize(depth_vis, (DISP_W, DISP_H))
    combined = np.hstack((color_disp, depth_disp))

    # キャッシュ用 dict
    frame_info = dict(
        frame_idx=frame_idx,
        combined=combined,
        depth_raw=_depth_raw.copy(),
        scale_x=_scale_x,
        scale_y=_scale_y,
        w_raw=_w_raw,
        h_raw=_h_raw,
    )
    history.append(frame_info)

    return combined

# -------------------------------------------------
# -------------------------------------------------
# 5. メインループ
# -------------------------------------------------

frame_idx = start_frame  # ★ 先に宣言して fetch_frame で参照可能にする

cv2.namedWindow("depth_view", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("depth_view", on_mouse)
combined = fetch_frame()
if combined is None:
    print("⚠ 最初のフレームが取得できませんでした。終了します。")
    pipeline.stop(); sys.exit(1)
cv2.resizeWindow("depth_view", combined.shape[1], combined.shape[0])

# キーコード集合
KEY_RIGHT = {2555904, 65363, 83}
KEY_LEFT  = {2424832, 65361, 81}
ESC = 27
("depth_view", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("depth_view", on_mouse)
combined = fetch_frame()
if combined is None:
    print("⚠ 最初のフレームが取得できませんでした。終了します。")
    pipeline.stop(); sys.exit(1)
cv2.resizeWindow("depth_view", combined.shape[1], combined.shape[0])

# キーコード集合
KEY_RIGHT = {2555904, 65363, 83}
KEY_LEFT  = {2424832, 65361, 81}
ESC = 27

frame_idx = start_frame  # ループ内で更新するので宣言位置は下でも OK

while True:
    cv2.imshow("depth_view", combined)
    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    if key == ESC:
        break

    # ---------- → キー : 次フレーム ---------------------------------
    elif key in KEY_RIGHT:
        frame_idx += 1
        new_frame = fetch_frame()
        if new_frame is not None:
            combined = new_frame
            print(f"▶ Frame {frame_idx} displayed (→)")

    # ---------- ← キー : 前フレーム ---------------------------------
    elif key in KEY_LEFT:
        if frame_idx == 0:
            continue
        # キャッシュに前フレームがあれば即座に取得
        if len(history) >= 2 and history[-2]['frame_idx'] == frame_idx - 1:
            history.pop()  # 現在フレームを捨てる
            prev = history[-1]
            combined = prev['combined']
            _depth_raw = prev['depth_raw']
            _scale_x, _scale_y = prev['scale_x'], prev['scale_y']
            _w_raw, _h_raw = prev['w_raw'], prev['h_raw']
            frame_idx -= 1
            print(f"▶ Frame {frame_idx} displayed from cache (←)")
            continue

        # キャッシュに無い → シーク＋flush
        frame_idx -= 1
        playback.pause()
        playback.seek(timedelta(microseconds=int(frame_idx * dt_us)))
        playback.resume()

        flushed = 0
        while flushed < 5 and pipeline.poll_for_frames():
            flushed += 1

        # キャッシュをクリア（現在より新しいフレームだけ削除）
        while history and history[-1]['frame_idx'] > frame_idx:
            history.pop()

        combined = fetch_frame()
        print(f"Frame {frame_idx} displayed (←, flushed {flushed})")

cv2.destroyAllWindows()
pipeline.stop()
