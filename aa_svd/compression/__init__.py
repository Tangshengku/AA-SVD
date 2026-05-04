import json
import logging
import os
from typing import Dict, Union

import torch
import torch.nn as nn
import wandb
from transformers import PreTrainedModel

from ..utils import get_device
from .compress import apply_compression_parallel
from .compressed_linear import CompressedLinear, QuantizedCompressedLinear
from .utils import replace_module

logger = logging.getLogger(__name__)


def get_modules_to_replace(model: nn.Module, config: Dict) -> list[str]:
    """Return linear module names selected by the compression config."""
    include_only = getattr(config, 'include_only', None)
    exclude_modules = getattr(config, 'exclude_only', None)

    if include_only is not None and exclude_modules is not None:
        raise ValueError(
            "Cannot specify both 'include_only' and 'exclude_modules'"
        )

    all_modules = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]

    if include_only is not None:
        if isinstance(include_only, str):
            include_only = [include_only]
        all_modules = [(n, m) for n, m in all_modules if n in include_only]
    elif exclude_modules is not None:
        if isinstance(exclude_modules, str):
            exclude_modules = [exclude_modules]
        all_modules = [
            (n, m) for n, m in all_modules if n not in exclude_modules
        ]

    return [name for name, _ in all_modules]


def load_complete_compressed_checkpoint(
    model: Union[PreTrainedModel, nn.Module],
    config: Dict,
) -> bool:
    """Load a complete per-module AA-SVD checkpoint if every selected module exists.

    Returns False for missing/partial checkpoints so callers can fall back to the
    normal compression path.
    """
    save_path = getattr(config, 'save_path', None)
    if save_path is None or not os.path.isdir(save_path):
        return False

    modules_to_replace = get_modules_to_replace(model, config)
    if not modules_to_replace:
        return False

    missing = []
    for module_name in modules_to_replace:
        module_dir = os.path.join(save_path, module_name.replace('.', '_'))
        has_plain_weights = (
            os.path.exists(os.path.join(module_dir, 'U.pt'))
            and os.path.exists(os.path.join(module_dir, 'V.pt'))
        )
        has_quantized_weights = (
            os.path.exists(os.path.join(module_dir, 'UK_q.pt'))
            and os.path.exists(os.path.join(module_dir, 'VK_q.pt'))
        )
        if not (has_plain_weights or has_quantized_weights):
            missing.append(module_name)

    if missing:
        logger.info(
            "Compressed checkpoint at %s is partial; %d/%d modules are missing.",
            save_path,
            len(missing),
            len(modules_to_replace),
        )
        return False

    loader = (
        QuantizedCompressedLinear
        if getattr(config, 'dobi_remapping', False)
        else CompressedLinear
    )
    dense_modules = dict(model.named_modules())
    for module_name in modules_to_replace:
        current_module = dense_modules[module_name]
        module_dir = os.path.join(save_path, module_name.replace('.', '_'))
        new_module = loader.from_path(
            module_dir,
            bias=current_module.bias,
        )
        new_module.to(
            device=current_module.weight.device,
            dtype=current_module.weight.dtype,
        )
        replace_module(model, module_name, new_module)

    for module_name, module in model.named_modules():
        if 'norm' not in module_name.lower():
            continue
        state_path = os.path.join(
            save_path,
            module_name.replace('.', '_'),
            'state_dict.pt',
        )
        if os.path.exists(state_path):
            module.load_state_dict(torch.load(state_path, map_location='cpu'))
            logger.info("Loaded norm module for %s from %s", module_name, state_path)

    logger.info(
        "Loaded complete AA-SVD checkpoint with %d modules from %s",
        len(modules_to_replace),
        save_path,
    )
    return True


def apply_compression(
    model: Union[PreTrainedModel, nn.Module],
    config: Dict,
    **kwargs
) -> nn.Module:
    """Apply SVD-based compression to all linear layers of a model."""

    model = model.to('cpu')
    torch.cuda.empty_cache()

    if hasattr(model, 'config'):
        use_cache_original = model.config.use_cache
        model.config.use_cache = False

    sub_method = getattr(config, 'sub_method', None)
    assert sub_method is not None, "sub_method must be specified in config"

    device = getattr(config, 'device', get_device())

    if sub_method == 'no-compress':
        return model.to(device)

    calibration_dataloader_train = kwargs.get('calibration_dataloader_train')
    calibration_dataloader_val = kwargs.get('calibration_dataloader_val')

    modules_to_replace = get_modules_to_replace(model, config)
    logger.info(f"Found {len(modules_to_replace)} linear layers for compression")

    if len(modules_to_replace) == 0:
        raise ValueError(
            "No linear layers found for compression. "
            "Please check your configuration."
        )

    target_param_ratio = getattr(config, 'target_param_ratio', 0.3)
    total_params_before = sum(p.numel() for p in model.parameters())

    # per-layer rank allocations
    rank_allocation_file_path = getattr(config, 'rank_allocation_file_path', None)
    if rank_allocation_file_path is not None:
        with open(rank_allocation_file_path, 'r') as f:
            allocations = json.load(f)
    else:
        allocations = {
            name: target_param_ratio for name in modules_to_replace
        }

    logger.info(f"Layerwise allocations: {json.dumps(allocations, indent=2)}")

    compressed_model = apply_compression_parallel(
        config, model, modules_to_replace,
        calibration_dataloader=calibration_dataloader_train,
        test_calibration_dataloader=calibration_dataloader_val,
        allocations=allocations,
        device=device,
    )

    del model
    torch.cuda.empty_cache()

    compressed_model = compressed_model.to(device)
    total_params_after = sum(p.numel() for p in compressed_model.parameters())
    ratio = total_params_after / total_params_before

    logger.info({
        'before': total_params_before,
        'after': total_params_after,
        'compression_ratio': ratio,
    })

    if wandb.run is not None:
        wandb.log({
            'params/before': total_params_before,
            'params/after': total_params_after,
            'params/compression_ratio': ratio,
        })

    if hasattr(compressed_model, 'config'):
        compressed_model.config.use_cache = use_cache_original

    return compressed_model
