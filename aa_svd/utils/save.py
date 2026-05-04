import json
import logging
import os
from typing import Any, Optional

from aa_svd.compression.compressed_linear import CompressedLinear

logger = logging.getLogger(__name__)


def save_hf_style_model(
    model: Any,
    tokenizer: Optional[Any],
    save_path: str,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
) -> None:
    """Save model, tokenizer, and AA-SVD compression metadata in a HF-style directory."""
    os.makedirs(save_path, exist_ok=True)

    compressed_modules = _compressed_module_metadata(model)
    if hasattr(model, "config"):
        model.config.aa_svd_compression = {
            "format": "compressed_linear_uv",
            "num_compressed_modules": len(compressed_modules),
            "modules": compressed_modules,
        }

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(
            save_path,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
    else:
        raise ValueError(
            "HF-style saving requires a model with save_pretrained()."
        )

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_path)

    metadata_path = os.path.join(save_path, "aa_svd_compression.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "format": "compressed_linear_uv",
                "note": (
                    "Weights are saved in Hugging Face directory layout. "
                    "CompressedLinear modules store low-rank U/V factors in "
                    "the model state dict and require AA-SVD code to reload."
                ),
                "num_compressed_modules": len(compressed_modules),
                "modules": compressed_modules,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved HF-style compressed model to {save_path}")


def _compressed_module_metadata(model: Any) -> list[dict[str, Any]]:
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, CompressedLinear):
            continue

        modules.append({
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "rank": module.rank,
            "bias": module.U.bias is not None,
            "compression_ratio": module._get_compression_ratio(),
        })
    return modules
