# GPUセットアップとクラウド

> 学習目的のCPUトレーニングはOK。本番のトレーニングにはGPUが必要だ。


## 学習目標

- `nvidia-smi` とPyTorchのCUDA APIを使ってローカルGPUの可用性を確認する
- 無料のクラウド実験のためにGoogle ColabをT4 GPUで設定する
- CPU対GPUの行列積をベンチマークし、速度向上を測定する
- fp16の目安ルールを使ってVRAMに収まる最大モデルサイズを見積もる

## 問題

フェーズ1〜3のほとんどのレッスンはCPUで問題なく動作する。しかし、CNN、トランスフォーマー、LLM（フェーズ4以降）のトレーニングを始めると、GPUアクセラレーションが必要になる。CPUで8時間かかるトレーニングはGPUなら10分で完了する。

選択肢は3つ: ローカルGPU、クラウドGPU、Google Colab（無料）。

## コンセプト

```
選択肢:

1. ローカルNVIDIA GPU
   コスト: $0（すでに持っている）
   セットアップ: CUDA + cuDNNのインストール
   最適: 日常利用、大規模データセット

2. Google Colab（無料枠）
   コスト: $0
   セットアップ: 不要
   最適: 短い実験、自宅にGPUがない場合

3. クラウドGPU（Lambda、RunPod、Vast.ai）
   コスト: $0.20〜2.00/時
   セットアップ: SSH + インストール
   最適: 本格的なトレーニング、大規模モデル
```

## 構築

### オプション1: ローカルNVIDIA GPU

GPUがあるか確認:

```bash
nvidia-smi
```

CUDAありのPyTorchをインストール:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### オプション2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com) にアクセス
2. ランタイム > ランタイムのタイプを変更 > T4 GPU
3. `!nvidia-smi` を実行して確認

このコースのノートブックをColabに直接アップロードできる。

### オプション3: クラウドGPU

Lambda Labs、RunPod、Vast.aiの場合:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPUがない場合も問題なし

ほとんどのレッスンはCPUで動作する。GPUが必要なレッスンにはその旨が書かれており、Colabのリンクも含まれる。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 構築: GPU対CPUベンチマーク

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 演習

1. 上記のベンチマークを実行し、CPUとGPUの時間を比較する
2. GPUがない場合は、Google Colabで実行して比較する
3. GPUメモリ量を確認し、収まる最大モデルサイズを見積もる（目安: fp16では1パラメーターあたり2バイト）

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| CUDA | 「GPUプログラミング」 | コードをGPU上で実行できるNVIDIAの並列コンピューティングプラットフォーム |
| VRAM | 「GPUメモリ」 | GPU上のビデオRAM。システムRAMとは別。モデルサイズを制限する。 |
| fp16 | 「半精度」 | 16ビット浮動小数点。fp32の半分のメモリで精度の損失は最小限 |
| テンソルコア | 「高速行列ハードウェア」 | 行列積に特化したGPUコア。通常のコアより4〜8倍高速 |
