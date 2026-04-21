import json
from pathlib import Path
from typing import Any

from .config import QuantizationMethod, W8A8Backend


def build_quantization_config(
    method: QuantizationMethod,
    dtype: str,
    w8a8_backend: W8A8Backend,
    model_path: str | Path | None = None,
) -> Any | None:
    if method == QuantizationMethod.BNB_4BIT:
        return _build_bnb_4bit_config(dtype)

    if method == QuantizationMethod.W8A8:
        return _build_w8a8_config(w8a8_backend, model_path)

    return None


def is_quantized_method(method: QuantizationMethod) -> bool:
    return method != QuantizationMethod.NONE


def get_serialized_quantization_config(model_path: str | Path | None) -> dict[str, Any] | None:
    if model_path is None:
        return None

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if "quantization_config" in config:
        return config["quantization_config"]

    if "compression_config" in config:
        return config["compression_config"]

    return None


def get_w8a8_backend(
    model_path: str | Path | None,
    requested_backend: W8A8Backend,
) -> W8A8Backend:
    if requested_backend != W8A8Backend.AUTO:
        return requested_backend

    serialized_config = get_serialized_quantization_config(model_path)
    if serialized_config is None:
        return W8A8Backend.AUTO

    quant_method = serialized_config.get("quant_method")
    if quant_method in {"compressed-tensors", "compressed_tensors"}:
        return W8A8Backend.COMPRESSED_TENSORS

    return W8A8Backend.AUTO


def requires_adapter_only_export(
    method: QuantizationMethod,
    w8a8_backend: W8A8Backend,
) -> bool:
    return (
        method == QuantizationMethod.W8A8
        and w8a8_backend == W8A8Backend.COMPRESSED_TENSORS
    )


def extract_weight_data(weight: Any) -> Any:
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ValueError(
                "bitsandbytes is required to dequantize 4-bit weights."
            ) from error

        return bnb.functional.dequantize_4bit(weight.data, quant_state)

    dequantize = getattr(weight, "dequantize", None)
    if callable(dequantize):
        return dequantize()

    data = getattr(weight, "data", None)
    dequantize = getattr(data, "dequantize", None)
    if callable(dequantize):
        return dequantize()

    tensor_impl = getattr(weight, "tensor_impl", None)
    dequantize = getattr(tensor_impl, "dequantize", None)
    if callable(dequantize):
        return dequantize()

    return weight


def _build_bnb_4bit_config(dtype: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise ValueError(
            "bitsandbytes 4-bit quantization requires PyTorch to be installed."
        ) from error

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as error:
        raise ValueError(
            "bitsandbytes 4-bit quantization requires a Transformers build with BitsAndBytesConfig."
        ) from error

    if dtype == "auto":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = getattr(torch, dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _build_w8a8_config(
    w8a8_backend: W8A8Backend,
    model_path: str | Path | None,
) -> Any | None:
    resolved_backend = get_w8a8_backend(model_path, w8a8_backend)

    if resolved_backend == W8A8Backend.AUTO:
        return None

    if resolved_backend == W8A8Backend.COMPRESSED_TENSORS:
        serialized_config = get_serialized_quantization_config(model_path)
        if serialized_config is None:
            raise ValueError(
                "W8A8 backend 'compressed_tensors' requires a checkpoint config with quantization metadata."
            )

        return _import_compressed_tensors_config().from_dict(serialized_config)

    if resolved_backend == W8A8Backend.QUANTO:
        try:
            from transformers import QuantoConfig
        except ImportError as error:
            raise ValueError(
                "W8A8 backend 'quanto' requires a Transformers build with QuantoConfig."
            ) from error

        return QuantoConfig(weights="int8", activations="int8")

    if resolved_backend == W8A8Backend.TORCHAO:
        try:
            from transformers import TorchAoConfig
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        except ImportError as error:
            raise ValueError(
                "W8A8 backend 'torchao' requires torchao to be installed."
            ) from error

        return TorchAoConfig(Int8DynamicActivationInt8WeightConfig())

    raise ValueError(f"Unsupported W8A8 backend: {resolved_backend}")


def _import_compressed_tensors_config():
    try:
        from transformers import CompressedTensorsConfig
    except ImportError as error:
        raise ValueError(
            "W8A8 backend 'compressed_tensors' requires a Transformers build with CompressedTensorsConfig."
        ) from error

    return CompressedTensorsConfig
