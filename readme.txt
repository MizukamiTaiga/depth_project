#仮想環境構築
py -3.10 -m venv venv310
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv310\Scripts\Activate
pip install -r requirements.txt

#アクティベート手順
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./〈仮想環境名〉/Scripts/activate
##確認用
pip list
##余計なライブラリ削除
pip uninstall opencv-python-headless -y
pip install opencv-python==4.11.0.86




#requirements.txt
numpy==2.2.6
opencv-python==4.11.0.86
pyrealsense2==2.55.1.6486
setuptools==65.5.0