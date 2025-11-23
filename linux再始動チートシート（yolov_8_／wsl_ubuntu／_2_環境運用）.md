# 0. まず「どこを開く？」（アクセスの仕方）

### Windows + WSL（Ubuntu）
- `Win + R` → `wsl` と入力で最後に使ったディストリを起動。
- 特定ディスト指定：`wsl -d Ubuntu-22.04`
- WSLホーム：`/home/<あなたのユーザ名>`
- Windows側のtドライブ：`/mnt/t/`
- Windowsの好きなフォルダを開く：`explorer.exe .`（現在ディレクトリをエクスプローラで開く）
- VSCode をこの場所で開く：`code .`

> ※ WSL の Ubuntu 上で作業します（今回 Docker は不要・削除済）。

---

# 1. プロジェクト起動ルーティン（毎回の最短手順）

```bash
# 1) Ubuntu を開く（上記のどれか）

# 2) プロジェクトフォルダへ移動（例）
cd ~/workspace/depth_project   # 例：あなたのパスに読み替え

# 3) 仮想環境（conda）を有効化
conda env list                 # 環境一覧を確認
conda activate yolo-env        # YOLO 用
# もしくは
# conda activate oid-env       # OIDv6（データDL）用

# 4) 依存の確認（初回またはrequirements更新時）
pip install -r requirements.txt

# 5) GPU と PyTorch の確認（任意）
python - << 'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
PY
```

> 以降、**学習・推論は `yolo-env`**、データ収集（Open Images）は **`oid-env`** で実行。

---

# 2. YOLOv8 の基本コマンド（Ultralytics）

> すべて `yolo-env` で実行。

### 2.1 画像/フォルダ/動画/カメラで推論
```bash
# 画像1枚
yolo predict model=yolov8n.pt source=sample.jpg save=True

# 画像フォルダ
yolo predict model=yolov8n.pt source=./images/ save=True

# Webカメラ（0番）
yolo predict model=yolov8n.pt source=0

# 信頼度やIoUなどしきい値
# 例: conf=0.35, iou=0.6, デバイスGPU:0
yolo predict model=yolov8s.pt source=./images/ conf=0.35 iou=0.6 device=0
```

### 2.2 自前データで学習（train）
```bash
# data.yaml に学習/検証パスとクラス名を定義しておく
# エポックと画像サイズを指定、学習ログは runs/ に保存

yolo task=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 device=0

# 再開（中断から続き）
yolo train resume=True
```

### 2.3 評価（validate）とバッチ推論（predict）
```bash
# 学習済み重みで検証
yolo val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640 device=0

# ディレクトリ一括推論
yolo predict model=runs/detect/train/weights/best.pt source=./test_images/ save=True
```

### 2.4 エクスポート（export）
```bash
# ONNX / TensorRT / OpenVINO などに変換（必要に応じて）
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```

---

# 3. データセット構成と `data.yaml` テンプレ

```
project_root/
  datasets/
    mydata/
      images/
        train/ ... .jpg
        val/   ... .jpg
      labels/
        train/ ... .txt   # YOLO書式: class x y w h (正規化)
        val/   ... .txt
  data.yaml
```

`data.yaml` 例：
```yaml
# data.yaml
path: ./datasets/mydata
train: images/train
val: images/val

nc: 2
names: ["crosswalk", "obstacle"]
```

> 既存のアノテーションが YOLO 形式なら、そのまま利用可能。クラス数（nc）と names を合わせる。

---

# 4. OIDv6（Open Images）でのデータ取得（`oid-env`）

> OIDv6 ToolKit は **別環境 `oid-env`** で運用。

```bash
# 1) 環境切替
conda activate oid-env

# 2) OIDv6 ツールキットのあるディレクトリへ
cd ~/workspace/open_images_toolkit   # 例

# 3) クラスを指定してダウンロード（例）
python main.py downloader --classes "Crosswalk" "Vending machine" \
  --type_csv train --limit 500 --multiclasses 1 --dataset ./oidv6

# 4) 取得後、YOLO 形式へ変換（必要に応じてツール側の変換スクリプトを使用）
# 5) datasets/mydata に整理して data.yaml を更新
```

> うまく行かない時は Python バージョン（3.10）と依存関係を確認。クラス名は公式表記で指定。

---

# 5. RealSense（任意：.bag→PNG 展開の最小例）

> 研究ノートの方針に従い、**カラーに深度をアライン**して扱う。

### 5.1 最小スクリプト（`scripts/export_frames.py`）
```python
import pyrealsense2 as rs
import cv2, os, sys

bag = sys.argv[1]
out = sys.argv[2]
os.makedirs(out, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag, repeat_playback=False)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

idx = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            break
        color_img = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
        depth_img = np.asanyarray(depth.get_data())
        cv2.imwrite(os.path.join(out, f"color_{idx:06d}.png"), color_img)
        cv2.imwrite(os.path.join(out, f"depth_{idx:06d}.png"), depth_img)
        idx += 1
except Exception:
    pass
finally:
    pipeline.stop()
```

実行例：
```bash
conda activate yolo-env
python scripts/export_frames.py ./bag/sample.bag ./captures/sample_export
```

---

# 6. よく使う Linux/WSL コマンド（忘れたとき用）

```bash
# ファイル操作
ls -al            # 詳細表示
pwd               # 現在地
cd path/to/dir    # 移動
mkdir -p dir/sub  # 階層作成
cp src dst        # コピー
mv src dst        # 移動/改名
rm file           # 削除（注意）
rm -r dir         # ディレクトリ削除（注意）

# 仮想環境
conda env list
conda activate yolo-env
conda activate oid-env
conda deactivate

# GPU/ドライバ
nvidia-smi

# エディタ
code .            # VSCode で開く
nano file.txt     # 端末内で手早く編集

# プロセス/長時間実行
htop              # リソース監視（入ってなければ: sudo apt install htop）
nohup <cmd> &     # 端末を閉じても続ける
```

---

# 7. 典型エラーと対処メモ
- **`command not found: yolo`** → `pip show ultralytics` で入っているか確認（`yolo-env` で `pip install -U ultralytics`）。
- **`ModuleNotFoundError: pyrealsense2`** → `pip install pyrealsense2`（Python 3.10で）。
- **CUDA が使われない** → `python -c "import torch; print(torch.cuda.is_available())"` が `False` のときは、`nvidia-smi` でドライバ/ハードを確認。WSL の GPU サポートが有効か（最新ドライバ/再起動）。
- **学習が遅い/落ちる** → `imgsz` の縮小、`batch` の調整、`yolov8n/s/m` など軽いモデルから。

---

# 8. 日々の運用マクロ（真似するだけ版）

```bash
# ── 推論だけ（学外デモなど）
wsl -d Ubuntu-22.04
cd ~/workspace/depth_project
conda activate yolo-env
yolo predict model=runs/detect/train/weights/best.pt source=./demo_images/ save=True

# ── 学習（GPU1枚）
wsl -d Ubuntu-22.04
cd ~/workspace/depth_project
conda activate yolo-env
yolo train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 device=0

# ── OIDv6 で補充データ収集
wsl -d Ubuntu-22.04
cd ~/workspace/open_images_toolkit
conda activate oid-env
python main.py downloader --classes "Crosswalk" --type_csv train --limit 500 --dataset ./oidv6
```

---

# 9. 次のアクション
1. Ubuntu を開き、プロジェクトに `cd`。
2. `conda activate yolo-env` → `pip install -r requirements.txt`（初回のみ）。
3. `yolo predict` をサンプルで1回実行して動作確認。
4. `data.yaml` を整備 → `yolo train` を小規模で試す。
5. 必要に応じて `oid-env` でデータ補充。

---

> 以降、必要に応じて本ドキュメントを更新していきます。疑問点が出たら、セクション番号を指定して質問してください（例：「5.1のスクリプトでエラー」など）。

