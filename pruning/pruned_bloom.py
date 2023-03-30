import math
import pickle as pkl

import torch 
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import LayerNorm

from transformers import BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomGelu, BloomMLP,  BloomModel
from transformers.models.bloom.configuration_bloom import BloomConfig

class PrunedBloomAttention(BloomAttention):
    def __init__(self, config: BloomConfig, self_atten_shapes):
        super().__init__(config)
        print("INIIT Pruned Bloom Attention")
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        
        query_key_value_output, query_key_value_input = self_atten_shapes["query_key_value.weight"]

        self.hidden_size = query_key_value_input
        self.num_heads = config.n_head
        self.head_dim = (query_key_value_output // 3) // self.num_heads
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != (query_key_value_output // 3):
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {query_key_value_output // 3} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        
        self.query_key_value = nn.Linear(query_key_value_input, query_key_value_output, bias=True)
        
        dense_output, dense_input = self_atten_shapes["dense.weight"]
        self.dense = nn.Linear(dense_input, dense_output)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
class PrunedBloomMLP(BloomMLP):
    def __init__(self, config: BloomConfig, mlp_shapes):
        super().__init__(config)
        print("INIT Pruned Bloom MLP")
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        
        dense_h_to_4h_output, dense_h_to_4h_input = mlp_shapes["dense_h_to_4h.weight"]
        self.dense_h_to_4h = nn.Linear(dense_h_to_4h_input, dense_h_to_4h_output)
        
        self.gelu_impl = BloomGelu()
        
        dense_4h_to_h_output, dense_4h_to_h_input = mlp_shapes["dense_4h_to_h.weight"]
        self.dense_4h_to_h = nn.Linear(dense_4h_to_h_input, dense_4h_to_h_output)
        self.hidden_dropout = config.hidden_dropout
    
class PrunedBloomBlock(BloomBlock):
    def __init__(self, config: BloomConfig, block_shapes):
        super().__init__(config)
        print("INIT pruned bloomn blocks")
        input_layer_norm_size = block_shapes["input_layernorm.weight"][0]
        self.input_layernorm = LayerNorm(input_layer_norm_size, eps=config.layer_norm_epsilon)
        
        self.num_heads = config.n_head
        self_atten_prefix = "self_attention."
        self_atten_shapes = {name.split(self_atten_prefix)[-1]: shape for name, shape in block_shapes.items() if self_atten_prefix in name}
        self.self_attention = PrunedBloomAttention(config, self_atten_shapes)
        
        post_atten_layer_norm_size = block_shapes["post_attention_layernorm.weight"][0]
        self.post_attention_layernorm = LayerNorm(post_atten_layer_norm_size, eps=config.layer_norm_epsilon)

        mlp_prefix = "mlp."
        mlp_shapes = {name.split(mlp_prefix)[-1]: shape for name, shape in block_shapes.items() if mlp_prefix in name}
        self.mlp = PrunedBloomMLP(config, mlp_shapes)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout
        
class PrunedBloomModel(BloomModel):
    def __init__(self, config: BloomConfig, state_dict_shapes):
        super().__init__(config)
        print("INIT pruned bloom model")

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Pruned Transformer blocks
        pruned_bloom_blocks = []
        for layer_num in range(config.num_hidden_layers):
            layer_prefix = f"transformer.h.{layer_num}."
            block_shapes = {name.split(layer_prefix)[-1]: shape for name, shape in state_dict_shapes.items() if layer_prefix in name}
            block = PrunedBloomBlock(config, block_shapes)
            pruned_bloom_blocks.append(block)
            
        self.h = nn.ModuleList(pruned_bloom_blocks)

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
class PrunedBloomForCausalLM(BloomForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: BloomConfig, state_dict_shapes_path):
        super().__init__(config)
        print("INIT pruned bloom model for causal LM")
        state_dict_shapes = pkl.load(open(state_dict_shapes_path, "rb"))
        print("Loaded state dict shapes")
        
        self.transformer = PrunedBloomModel(config, state_dict_shapes)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
