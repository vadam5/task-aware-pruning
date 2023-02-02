from typing import Dict, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.generation.utils import SampleDecoderOnlyOutput

class BloomOutputForNodeAttribution(BaseModelOutputWithPastAndCrossAttentions):
    activations: Dict[str, Dict] = None
    
class CausalLMOutputForNodeAttribution(CausalLMOutputWithCrossAttentions):
    activations: Dict[str, Dict] = None
    
class SampleDecoderOnlyOutputForNodeAttribution(SampleDecoderOnlyOutput):
    activations: Tuple[Dict[str, Dict]] = None