from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from .llama_adapter import LlamaLayerAdapter, LlamaHeadAdapter, LlamaModelAdapter
from .qwen2_adapter import Qwen2LayerAdapter, Qwen2HeadAdapter, Qwen2ModelAdapter

MODEL_ADAPTER_REGISTRY = {
    LlamaForCausalLM: LlamaModelAdapter,
    Qwen2ForCausalLM: Qwen2ModelAdapter,
}

try:
    from transformers import MistralForCausalLM
except ImportError:
    try:
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    except ImportError:
        MistralForCausalLM = None

if MistralForCausalLM is not None:
    MODEL_ADAPTER_REGISTRY[MistralForCausalLM] = LlamaModelAdapter

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except ImportError:
        Qwen3ForCausalLM = None

if Qwen3ForCausalLM is not None:
    MODEL_ADAPTER_REGISTRY[Qwen3ForCausalLM] = Qwen2ModelAdapter

__all__ = [
    "LlamaLayerAdapter",
    "LlamaHeadAdapter",
    "LlamaModelAdapter",
    "Qwen2LayerAdapter",
    "Qwen2HeadAdapter",
    "Qwen2ModelAdapter",
    "MODEL_ADAPTER_REGISTRY",
]
