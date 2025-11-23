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

## 環境構築
推奨環境: Windows 11 + Python 3.10

1. 仮想環境の作成と有効化
   py -3.10 -m venv venv310
   .\venv310\Scripts\Activate

2. ライブラリのインストール
   pip install -r requirements.txt

## 実行方法
システムを起動するには、以下のコマンドを実行します。
python src/main.py --map map.json --model yolov8n.pt

終了するには、表示されたウィンドウ上で 'q' を押してください。

## テストの実行
モックデータを用いた動作確認テストを実行するには:
python tests/test_system_mock.py

## 補足
- 詳細な研究ガイドラインは `real_sense_research_guide.md` を参照してください。
- 開発環境の詳細は `README_for_CustomGPT.md` を参照してください。