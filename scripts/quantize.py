"""
Dynamic per-channel INT8 quantization of WD-ViT-Tagger-v3.

Run with the local venv that has onnxruntime + onnx + sympy installed:

    /tmp/ort-venv/bin/python scripts/quantize.py

Outputs `model-int8.onnx` next to `model.onnx`. Per-channel weight
quantization is used for Conv/MatMul to keep top-tag precision on the
ViT head — for typical RP-relevant tags (smile / long_hair / nude /
indoor / etc.) the difference vs. FP32 is invisible after the 0.35
confidence threshold.
"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "model.onnx"
DST = HERE / "model-int8.onnx"

if not SRC.exists():
    raise SystemExit(f"missing {SRC} — download model.onnx first")

print(f"src: {SRC}  ({SRC.stat().st_size / 1e6:.1f} MB)")
print(f"dst: {DST}")
print("quantizing (per-channel QInt8) — this takes ~30 seconds…")

quantize_dynamic(
    model_input=str(SRC),
    model_output=str(DST),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)

print(f"done. INT8 size: {DST.stat().st_size / 1e6:.1f} MB "
      f"({DST.stat().st_size / SRC.stat().st_size * 100:.1f}% of FP32)")
