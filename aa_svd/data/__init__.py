import logging
import os
from typing import Any, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def create_datasets(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Create train/val datasets based on configuration."""
    dataset_type = config.get('type')
    if dataset_type == 'huggingface':
        return _load_huggingface_dataset(config, tokenizer)
    elif dataset_type == 'huggingface_mixture':
        return _load_huggingface_mixture_dataset(config, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_huggingface_dataset(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a HuggingFace dataset and cache it to disk as binary shards."""
    block_size = config.get('block_size', 512)
    num_samples = config.get('num_samples', None)
    num_samples_val = config.get('num_samples_val', num_samples)
    sampling = config.get('sampling', 'random')
    seed = config.get('seed', 42)

    data_path = os.path.join(
        config.get('data_path'),
        config.get('name'),
        config.get('subset'),
        tokenizer.name_or_path,
        config.get('task_type'),
    )

    if os.path.exists(data_path):
        try:
            logger.info(f"Loading dataset from {data_path}")
            from .iterable_text_dataset import TextCalibrationDataset
            train_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples, sampling=sampling,
                split='train', seed=seed,
            )
            val_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples_val, sampling=sampling,
                split='val', seed=seed,
            )
            return train_dataset, val_dataset
        except Exception as e:
            logger.warning(
                f"Failed to load dataset from disk: {e}. "
                "Will download from HuggingFace and preprocess."
            )

    name = config.get('name')
    subset = config.get('subset')
    logger.info(f"Loading subset={subset} for dataset={name} from HuggingFace")

    if name == "c4":
        url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-train.00000-of-01024.json.gz"
        )
        dataset = load_dataset("json", data_files=url)
        val_url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-validation.00000-of-00008.json.gz"
        )
        dataset["val"] = load_dataset("json", data_files=val_url)["train"]
    else:
        dataset = load_dataset(name, subset, num_proc=8)

    if hasattr(config, 'preprocessing'):
        from .utils import _apply_preprocessing
        dataset = _apply_preprocessing(
            name, dataset, config.preprocessing, tokenizer
        )

    from .utils import save_dataset_to_disk
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logger.info(f"Saving dataset to {data_path}")
    save_dataset_to_disk(dataset, data_path, tokenizer=tokenizer)

    from .iterable_text_dataset import TextCalibrationDataset
    train_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples, sampling=sampling,
        split='train', seed=seed,
    )
    val_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples_val, sampling=sampling,
        split='val', seed=seed,
    )
    return train_dataset, val_dataset


def _load_huggingface_mixture_dataset(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Load several HuggingFace datasets and cache one mixed token stream."""
    block_size = config.get('block_size', 512)
    num_samples = config.get('num_samples', None)
    num_samples_val = config.get('num_samples_val', num_samples)
    sampling = config.get('sampling', 'random')
    seed = config.get('seed', 42)

    tokenizer_name = tokenizer.name_or_path if tokenizer is not None else "no_tokenizer"
    data_path = os.path.join(
        config.get('data_path'),
        config.get('name'),
        config.get('subset', 'default'),
        tokenizer_name,
        config.get('task_type'),
    )

    if os.path.exists(data_path):
        try:
            logger.info(f"Loading dataset mixture from {data_path}")
            from .iterable_text_dataset import TextCalibrationDataset
            train_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples, sampling=sampling,
                split='train', seed=seed,
            )
            val_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples_val, sampling=sampling,
                split='val', seed=seed,
            )
            return train_dataset, val_dataset
        except Exception as e:
            logger.warning(
                f"Failed to load dataset mixture from disk: {e}. "
                "Will download from HuggingFace and preprocess."
            )

    train_parts = []
    val_parts = []
    for dataset_cfg in config.get('datasets'):
        name = dataset_cfg.get('name')
        subset = dataset_cfg.get('subset', None)
        split = dataset_cfg.get('split', 'train')
        val_split = dataset_cfg.get('validation_split', None)
        logger.info(f"Loading mixture component dataset={name}, subset={subset}, split={split}")

        load_kwargs = {}
        if subset is not None:
            load_args = (name, subset)
        else:
            load_args = (name,)
        dataset = load_dataset(*load_args, **load_kwargs)

        train_split = dataset[split] if isinstance(dataset, DatasetDict) else dataset
        val_source = None
        if isinstance(dataset, DatasetDict):
            if val_split is not None and val_split in dataset:
                val_source = dataset[val_split]
            elif 'validation' in dataset:
                val_source = dataset['validation']
            elif 'val' in dataset:
                val_source = dataset['val']
            elif 'test' in dataset:
                val_source = dataset['test']
        if val_source is None:
            val_source = train_split

        train_parts.append(_normalize_text_dataset(train_split, tokenizer, dataset_cfg))
        val_parts.append(_normalize_text_dataset(val_source, tokenizer, dataset_cfg))

    dataset = DatasetDict({
        'train': concatenate_datasets(train_parts),
        'val': concatenate_datasets(val_parts),
    })

    if hasattr(config, 'preprocessing'):
        from .utils import _apply_preprocessing
        dataset = _apply_preprocessing(
            config.get('name'), dataset, config.preprocessing, tokenizer
        )

    from .utils import save_dataset_to_disk
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logger.info(f"Saving dataset mixture to {data_path}")
    save_dataset_to_disk(dataset, data_path, tokenizer=tokenizer)

    from .iterable_text_dataset import TextCalibrationDataset
    train_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples, sampling=sampling,
        split='train', seed=seed,
    )
    val_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples_val, sampling=sampling,
        split='val', seed=seed,
    )
    return train_dataset, val_dataset


def _normalize_text_dataset(dataset: Dataset, tokenizer: Optional[Any], dataset_cfg: Any) -> Dataset:
    """Convert common plain-text, instruction, and chat rows into a text column."""
    text_field = dataset_cfg.get('text_field', None)
    text_template = dataset_cfg.get('text_template', None)

    def _row_to_text(row):
        if text_field is not None and text_field in row:
            return {'text': _stringify(row[text_field])}
        if text_template is not None:
            return {'text': text_template.format(**{k: _stringify(v) for k, v in row.items()})}
        if 'text' in row:
            return {'text': _stringify(row['text'])}
        if 'messages' in row:
            return {'text': _messages_to_text(row['messages'], tokenizer)}
        if 'conversations' in row:
            return {'text': _messages_to_text(row['conversations'], tokenizer)}

        prompt_parts = []
        for key in ['system', 'instruction', 'input', 'question', 'prompt']:
            value = row.get(key)
            if value:
                prompt_parts.append(_stringify(value))
        answer_parts = []
        for key in ['output', 'response', 'answer', 'completion']:
            value = row.get(key)
            if value:
                answer_parts.append(_stringify(value))
        if prompt_parts or answer_parts:
            return {'text': "\n\n".join(prompt_parts + answer_parts)}

        for value in row.values():
            if isinstance(value, str) and value.strip():
                return {'text': value}
        return {'text': ""}

    remove_columns = dataset.column_names
    dataset = dataset.map(_row_to_text, remove_columns=remove_columns)
    return dataset.filter(lambda x: len(x['text']) > 0)


def _messages_to_text(messages, tokenizer: Optional[Any]) -> str:
    if isinstance(messages, str):
        return messages

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    rendered = []
    for message in messages or []:
        if not isinstance(message, dict):
            rendered.append(_stringify(message))
            continue
        role = message.get('role') or message.get('from') or 'user'
        content = message.get('content') or message.get('value') or ''
        rendered.append(f"{role}: {_stringify(content)}")
    return "\n".join(rendered)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    return str(value)


def get_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
