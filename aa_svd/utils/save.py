import json
import logging
import os
from typing import Any, Optional

import torch
from torch import nn

from aa_svd.compression.compressed_linear import CompressedLinear
from aa_svd.compression.utils import replace_module

logger = logging.getLogger(__name__)


def save_hf_style_model(
    model: Any,
    tokenizer: Optional[Any],
    save_path: str,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    materialize_compressed: bool = False,
) -> None:
    """Save model and tokenizer in a HF-style directory.

    When materialize_compressed is true, AA-SVD CompressedLinear modules are
    converted back to regular dense Linear modules before saving so plain
    transformers AutoModel loaders can consume the checkpoint directly.
    """
    os.makedirs(save_path, exist_ok=True)

    compressed_modules = _compressed_module_metadata(model)
    if materialize_compressed:
        _materialize_compressed_linears(model)
        compressed_modules = []
        if hasattr(model, "config") and hasattr(model.config, "aa_svd_compression"):
            delattr(model.config, "aa_svd_compression")
    elif hasattr(model, "config"):
        model.config.aa_svd_compression = {
            "format": "compressed_linear_uv",
            "num_compressed_modules": len(compressed_modules),
            "modules": compressed_modules,
        }

    if not hasattr(model, "save_pretrained"):
        raise ValueError(
            "HF-style saving requires a model with save_pretrained()."
        )

    model.save_pretrained(
        save_path,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_path)

    if materialize_compressed:
        metadata_path = os.path.join(save_path, "aa_svd_compression.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        logger.info(f"Saved dense HF model to {save_path}")
        return

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


def _materialize_compressed_linears(model: Any) -> None:
    compressed_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CompressedLinear)
    ]

    for name, module in compressed_modules:
        dense_module = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.U.bias is not None,
            device=module.U.weight.device,
            dtype=module.U.weight.dtype,
        )
        with torch.no_grad():
            dense_module.weight.copy_(module.get_recon_weight())
            if module.U.bias is not None:
                dense_module.bias.copy_(module.U.bias)
        replace_module(model, name, dense_module)


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
