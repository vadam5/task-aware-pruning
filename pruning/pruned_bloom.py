import math
import pickle as pkl

import torch 
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import LayerNorm

from transformers import BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomGelu, BloomMLP,  BloomModel
from transformers.models.bloom.configuration_bloom import BloomConfig

# class PrunedDropoutAdd:   
#     def pruned_dropout_add(self, x, residual, prob, training) -> torch.Tensor:
#         """
#         Need custom dropout add function to make pruned residual connections work. 
#         """
#         # Need to truncate or pad the residual to align with the out dimension
        
#         out = F.dropout(x, p=prob, training=training)
#         out = residual + out
#         return out

class PrunedBloomAttention(BloomAttention):
    def __init__(self, config: BloomConfig, self_atten_shapes):
        super().__init__(config)
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
        
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     residual: torch.Tensor,
    #     alibi: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     layer_past = None,
    #     head_mask = None,
    #     use_cache: bool = False,
    #     output_attentions: bool = False,
    # ):
    #     fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    #     # 3 x [batch_size, seq_length, num_heads, head_dim]
    #     (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    #     batch_size, q_length, _, _ = query_layer.shape

    #     query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    #     key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
    #     value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    #     if layer_past is not None:
    #         past_key, past_value = layer_past
    #         # concatenate along seq_length dimension:
    #         #  - key: [batch_size * self.num_heads, head_dim, kv_length]
    #         #  - value: [batch_size * self.num_heads, kv_length, head_dim]
    #         key_layer = torch.cat((past_key, key_layer), dim=2)
    #         value_layer = torch.cat((past_value, value_layer), dim=1)

    #     _, _, kv_length = key_layer.shape

    #     if use_cache is True:
    #         present = (key_layer, value_layer)
    #     else:
    #         present = None

    #     # [batch_size * num_heads, q_length, kv_length]
    #     # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
    #     matmul_result = alibi.baddbmm(
    #         batch1=query_layer,
    #         batch2=key_layer,
    #         beta=self.beta,
    #         alpha=self.inv_norm_factor,
    #     )

    #     # change view to [batch_size, num_heads, q_length, kv_length]
    #     attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

    #     # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
    #     input_dtype = attention_scores.dtype
    #     # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
    #     if input_dtype == torch.float16:
    #         attention_scores = attention_scores.to(torch.float)
    #     attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
    #     attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

    #     # [batch_size, num_heads, q_length, kv_length]
    #     attention_probs = self.attention_dropout(attention_probs)

    #     if head_mask is not None:
    #         attention_probs = attention_probs * head_mask

    #     # change view [batch_size x num_heads, q_length, kv_length]
    #     attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

    #     # matmul: [batch_size * num_heads, q_length, head_dim]
    #     context_layer = torch.bmm(attention_probs_reshaped, value_layer)

    #     # change view [batch_size, num_heads, q_length, head_dim]
    #     context_layer = self._merge_heads(context_layer)

    #     # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    #     if self.pretraining_tp > 1 and self.slow_but_exact:
    #         slices = self.hidden_size / self.pretraining_tp
    #         output_tensor = torch.zeros_like(context_layer)
    #         for i in range(self.pretraining_tp):
    #             output_tensor = output_tensor + F.linear(
    #                 context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
    #                 self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
    #             )
    #     else:
    #         output_tensor = self.dense(context_layer)

    #     output_tensor = self.pruned_dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    #     outputs = (output_tensor, present)
    #     if output_attentions:
    #         outputs += (attention_probs,)

    #     return outputs
    
class PrunedBloomMLP(BloomMLP):
    def __init__(self, config: BloomConfig, mlp_shapes):
        super().__init__(config)
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        
        dense_h_to_4h_output, dense_h_to_4h_input = mlp_shapes["dense_h_to_4h.weight"]
        self.dense_h_to_4h = nn.Linear(dense_h_to_4h_input, dense_h_to_4h_output)
        
        self.gelu_impl = BloomGelu()
        
        dense_4h_to_h_output, dense_4h_to_h_input = mlp_shapes["dense_4h_to_h.weight"]
        self.dense_4h_to_h = nn.Linear(dense_4h_to_h_input, dense_4h_to_h_output)
        self.hidden_dropout = config.hidden_dropout
        
    #     self.residual_pruned_mapping = 
        
    # def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    #     hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

    #     if self.pretraining_tp > 1 and self.slow_but_exact:
    #         intermediate_output = torch.zeros_like(residual)
    #         slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
    #         for i in range(self.pretraining_tp):
    #             intermediate_output = intermediate_output + F.linear(
    #                 hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
    #                 self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
    #             )
    #     else:
    #         intermediate_output = self.dense_4h_to_h(hidden_states)

    #     output = self.pruned_dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

    #     return output
    
class PrunedBloomBlock(BloomBlock):
    def __init__(self, config: BloomConfig, block_shapes):
        super().__init__(config)
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
        state_dict_shapes = pkl.load(open(state_dict_shapes_path, "rb"))
        
        self.transformer = PrunedBloomModel(config, state_dict_shapes)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()