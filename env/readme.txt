# 視覚障がい者向けマルチモーダルナビゲーションシステム

RealSense (RGB-D) と ReSpeaker (音響) を用いた、視覚障がい者向けのナビゲーションシステムです。
YOLOv8によるランドマーク検出と、自作マップを用いた自己位置推定を行います。

# 仮想環境種類
- yolo-env → 学習・推論・RealSense解析
- oid-env → Open Images データ収集用
- depth_project → main環境
- 各環境更新後は export すること


## ディレクトリ構成
depth_project/
├── src/
│   ├── main.py              # メイン実行ファイル
│   ├── sensors/             # センサドライバ (RealSense, ReSpeaker)
│   ├── vision/              # 画像処理 (YOLOv8 + Depth)
│   ├── audio/               # 音響処理 (DOA)
│   ├── map/                 # マップ管理
│   └── navigation/          # 自己位置推定ロジック
├── tests/                   # テストコード
├── map.json                 # ランドマークマップ定義ファイル
└── requirements.txt         # 依存ライブラリ一覧

## テストの実行
モックデータを用いた動作確認テストを実行するには:
python tests/test_system_mock.py

# 視覚障がい者向けマルチモーダルナビゲーションシステム

RealSense (RGB-D) と ReSpeaker (音響) を用いた、視覚障がい者向けのナビゲーションシステムです。
YOLOv8によるランドマーク検出と、自作マップを用いた自己位置推定を行います。

## ディレクトリ構成
depth_project/
├── src/
│   ├── main.py              # メイン実行ファイル
│   ├── sensors/             # センサドライバ (RealSense, ReSpeaker)
│   ├── vision/              # 画像処理 (YOLOv8 + Depth)
│   ├── audio/               # 音響処理 (DOA)
│   ├── map/                 # マップ管理
│   └── navigation/          # 自己位置推定ロジック
├── tests/                   # テストコード
├── map.json                 # ランドマークマップ定義ファイル
└── requirements.txt         # 依存ライブラリ一覧

## 環境構築 (Condaの場合)
Anaconda / Miniconda を使用する場合:

1. 環境の作成
   conda env create -f env/environment_depth_project.yml

2. 環境の有効化
   conda activate depth_project

## 実行方法
システムを起動するには、以下のコマンドを実行します。
python src/main.py --map map.json --model yolov8n.pt

終了するには、表示されたウィンドウ上で 'q' を押してください。

## テストの実行
モックデータを用いた動作確認テストを実行するには:
python tests/test_system_mock.py

## 補足
- 詳細な研究ガイドラインは `real_sense_research_guide.md` を参照してください。

