# Finger-Multi-EMG-Inference

## 概要
指を動かした時の筋電位波形の学習プログラム

## コードについて
- EMGTrain.py
  - 筋電位波形学習プログラム
- EMGTest.py
  - 筋電位波形学習済みモデルのテストプログラム
- EMGTransferTrain.py
  - 5本指の屈伸動作分類モデルを用いたキー分類モデルへの転移学習プログラム
- EMGQuantization.py
  - FCモデルへの量子化実行プログラム
- EMGTimeMeasure.py
  - 筋電位学習済みモデルの推論時間測定プログラム
- EMGFCQuantizationPred.py
  - 量子化を施した筋電位学習済みFCモデルの推論時間測定プログラム
- EMGLSTMQuantizationPred.py
  - 量子化を施した筋電位学習済みLSTMモデルの推論時間測定プログラム
- settings.py
  - 学習・テストに使う設定を記述したコード
- model.py
  - 学習するモデルを記述したコード
- quantize_model.py
  - 量子化に対応したモデルを記述したコード
- dataset.py
  - データセットの読み込みを行うコード
- training.py
  - 学習・検証・テストを実際に行うコード

## 実行方法
基本的に、EMG**.pyを実行することで、学習やテストを行うことができる。\
`python3 EMG**.py --help`を実行するとusageが出てくるため、そちらを参照願いたい。