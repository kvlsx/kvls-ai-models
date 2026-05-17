"""
Static INT8 quantization of WD-ViT-Tagger-v3 with a small calibration set.

⚠ EXPERIMENTAL — DOES NOT WORK for this model. Kept for reproducibility.

Two failure modes observed on this ViT (run against the FP32 reference):
  - QOperator + MinMax calibration: model collapses; every image yields the
    same argmax cluster regardless of content (Pearson corr ≈ 0.1).
  - QDQ + Entropy calibration: 16 GB RAM is exhausted during the calibration
    pass over 100 images and the kernel kills the process (SIGKILL 137).

In production we ship the *dynamic* quantization variant (`quantize.py`)
which retains correlation 0.81–0.96 with FP32 and the same 97 MB footprint.
The script below is left here so the failure is reproducible and any future
fix (e.g. per-tensor activations, op-level skip-list for attention/softmax,
or a smaller calibration set) can be evaluated against it.

Run inside the local venv (onnxruntime + onnx + sympy + Pillow + numpy):

    /tmp/ort-venv/bin/python scripts/quantize_static.py

Inputs (must exist next to this script's parent dir):
    model.onnx            (FP32, ≈361 MB)
    calibration/*.jpg     (≈100 generic 448×448 JPEGs from Picsum;
                           any natural images work — variety > subject)

Output:
    model-int8.onnx       (≈90 MB, replaces the dynamic-quant variant)
"""
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "model.onnx"
SRC_PREP = HERE / "model.preprocessed.onnx"
DST = HERE / "model-int8.onnx"
CALIB_DIR = HERE / "calibration"

INPUT_EDGE = 448


def fit_to_square_bgr(img: Image.Image) -> np.ndarray:
    """Same preprocessing as `WdTaggerModel.fitToSquare` on Android / iOS:
    pad to square with white, resize to 448×448, return BGR float32 HWC."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long = max(w, h)
    square = Image.new("RGB", (long, long), (255, 255, 255))
    square.paste(img, ((long - w) // 2, (long - h) // 2))
    square = square.resize((INPUT_EDGE, INPUT_EDGE), Image.BILINEAR)
    # PIL gives RGB; model expects BGR.
    arr = np.asarray(square, dtype=np.float32)[..., ::-1].copy()
    return arr  # (448, 448, 3) BGR float32 in [0, 255]


class PicsumCalibrationReader(CalibrationDataReader):
    """Yields one preprocessed batch per JPEG in `calibration/`."""

    def __init__(self, input_name: str, files):
        self.input_name = input_name
        self._iter = iter(files)

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        arr = fit_to_square_bgr(Image.open(path))
        return {self.input_name: arr[np.newaxis, ...]}  # batch dim


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    files = sorted(CALIB_DIR.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no calibration images in {CALIB_DIR}")

    # 1) Pre-process the model graph: symbolic shape inference + folding,
    #    needed for ViTs so the quantizer can see Gemm/MatMul tensor shapes.
    print(f"pre-processing {SRC.name} → {SRC_PREP.name} …")
    quant_pre_process(str(SRC), str(SRC_PREP), skip_symbolic_shape=False)

    # Discover the input name from the (preprocessed) model.
    sess = ort.InferenceSession(str(SRC_PREP), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"input tensor name: {input_name}")
    del sess

    reader = PicsumCalibrationReader(input_name, files)
    print(f"calibrating on {len(files)} images …")

    # 2) Static quantization. Per-channel symmetric weights (QInt8) +
    #    asymmetric activations (QUInt8) is the standard recipe for ViTs.
    # QDQ format (QuantizeLinear/DequantizeLinear ops around target nodes)
    # rather than QOperator (whole-op quantized kernels). QDQ is the modern
    # recommended path for transformers — keeps softmax/layer-norm and other
    # numerically sensitive ops in fp32 and only quantizes Conv/MatMul where
    # it's safe. QOperator broke this ViT entirely (all photos collapsed to
    # the same argmax cluster — verified against the FP32 reference).
    #
    # Entropy calibration tracks the activation distribution's tail better
    # than naive MinMax for transformer activations.
    quantize_static(
        model_input=str(SRC_PREP),
        model_output=str(DST),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy,
    )
    SRC_PREP.unlink(missing_ok=True)

    size_src = SRC.stat().st_size / 1e6
    size_dst = DST.stat().st_size / 1e6
    print(f"done. FP32 {size_src:.1f} MB → static-INT8 {size_dst:.1f} MB "
          f"({size_dst / size_src * 100:.1f}%)")


if __name__ == "__main__":
    main()
