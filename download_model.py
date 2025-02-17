import whisperx
import torch

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # 尝试加载中文对齐模型
    model_a, metadata = whisperx.load_align_model(language_code="zh", device=device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
