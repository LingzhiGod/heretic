import json

import pytest

from heretic.config import QuantizationMethod, Settings, W8A8Backend
from heretic.quantization import (
    build_quantization_config,
    extract_weight_data,
    get_serialized_quantization_config,
    get_w8a8_backend,
    requires_adapter_only_export,
)


class FakeWeight:
    def __init__(self):
        self.called = False

    def dequantize(self):
        self.called = True
        return "dequantized"


def test_settings_accept_w8a8_quantization(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pytest"])

    settings = Settings(model="dummy", quantization="w8a8", w8a8_backend="auto")

    assert settings.quantization is QuantizationMethod.W8A8
    assert settings.w8a8_backend is W8A8Backend.AUTO


def test_w8a8_auto_backend_uses_checkpoint_quantization():
    config = build_quantization_config(
        QuantizationMethod.W8A8,
        "bfloat16",
        W8A8Backend.AUTO,
    )

    assert config is None


def test_w8a8_quanto_backend_builds_w8a8_config():
    config = build_quantization_config(
        QuantizationMethod.W8A8,
        "bfloat16",
        W8A8Backend.QUANTO,
    )

    assert config.weights == "int8"
    assert config.activations == "int8"


def test_extract_weight_data_uses_dequantize_when_available():
    weight = FakeWeight()

    result = extract_weight_data(weight)

    assert result == "dequantized"
    assert weight.called is True


def test_reads_legacy_compression_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "compression_config": {
                    "format": "int-quantized",
                    "quant_method": "compressed-tensors",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        ),
        encoding="utf-8",
    )

    config = get_serialized_quantization_config(tmp_path)

    assert config == {
        "format": "int-quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"targets": ["Linear"]}},
    }


def test_auto_backend_detects_compressed_tensors(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "compression_config": {
                    "format": "int-quantized",
                    "quant_method": "compressed-tensors",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        ),
        encoding="utf-8",
    )

    backend = get_w8a8_backend(tmp_path, W8A8Backend.AUTO)

    assert backend is W8A8Backend.COMPRESSED_TENSORS


def test_compressed_tensors_requires_adapter_only_export():
    assert (
        requires_adapter_only_export(
            QuantizationMethod.W8A8,
            W8A8Backend.COMPRESSED_TENSORS,
        )
        is True
    )


def test_explicit_compressed_tensors_backend_builds_quant_config(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "compression_config": {
                    "format": "int-quantized",
                    "quant_method": "compressed-tensors",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeCompressedTensorsConfig:
        @classmethod
        def from_dict(cls, value):
            return {"loaded": value}

    def fake_import():
        return FakeCompressedTensorsConfig

    monkeypatch.setattr(
        "heretic.quantization._import_compressed_tensors_config",
        fake_import,
    )

    config = build_quantization_config(
        QuantizationMethod.W8A8,
        "bfloat16",
        W8A8Backend.COMPRESSED_TENSORS,
        model_path=tmp_path,
    )

    assert config == {
        "loaded": {
            "format": "int-quantized",
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"targets": ["Linear"]}},
        }
    }
