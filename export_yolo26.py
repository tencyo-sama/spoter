"""
export_yolo26.py
----------------
YOLO26 Nano (Native End-to-End / NMS-free) を ONNX Runtime Web 向けにエクスポートするスクリプト。

[実行方法]
  # uv を使用して依存パッケージをインストール・実行
  uv run --with ultralytics --with onnx --with onnxruntime python export_yolo26.py

出力ファイル:
  yolo26n.onnx (640x640, End-to-End NMS-free 出力 [1, 300, 6] : [x1, y1, x2, y2, score, class_id])
"""

import sys
from ultralytics import YOLO

def export_yolo26():
    model_name = "yolo26n.pt"
    print(f"[+] Loading model: {model_name}...")
    
    # YOLO26 Nano モデルの読み込み
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[!] Warning: Could not find '{model_name}'. Attempting fallback to standard nano or downloading.")
        # yolo26n が利用できない環境向けにフォールバック（例: yolo11n.pt や yolo8n.pt 等でも同等APIでエクスポート可能）
        model = YOLO("yolo11n.pt")

    print("[+] Exporting to ONNX for Web (ONNX Runtime Web: WebGPU / WASM)...")
    
    # Web向けONNXエクスポート設定:
    # - format: onnx
    # - imgsz: 640
    # - dynamic: False (WebGPUでの最適化とメモリ事前確保のため固定形状を推奨)
    # - simplify: ONNXグラフの最適化・軽量化
    # - opset: 17 (最新WebGPU/WASM環境で広くサポートされる安定版)
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=17
    )
    
    print(f"[✓] Export successful! Model exported to: {exported_path}")
    print("[*] Place the exported ONNX file as 'yolo26n.onnx' in the web app root directory.")

if __name__ == "__main__":
    export_yolo26()
