import math
import pickle as pkl

import torch 
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import LayerNorm

from transformers import BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomGelu, BloomMLP,  BloomModel
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import dropout_add, build_alibi_tensor

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class PrunedBloomAttention(BloomAttention):
    def __init__(self, config: BloomConfig, self_atten_shapes, num_heads):
        super().__init__(config)
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        
        query_key_value_output, query_key_value_input = self_atten_shapes["query_key_value.weight"]

        self.hidden_size = query_key_value_input
        self.num_heads = num_heads
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
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past = None,
        head_mask = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

    
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
    
class PrunedBloomBlock(BloomBlock):
    def __init__(self, config: BloomConfig, block_shapes, num_heads):
        super().__init__(config)
        input_layer_norm_size = block_shapes["input_layernorm.weight"][0]
        self.input_layernorm = LayerNorm(input_layer_norm_size, eps=config.layer_norm_epsilon)
        
        self.num_heads = num_heads
        self_atten_prefix = "self_attention."
        self_atten_shapes = {name.split(self_atten_prefix)[-1]: shape for name, shape in block_shapes.items() if self_atten_prefix in name}
        self.self_attention = PrunedBloomAttention(config, self_atten_shapes, num_heads)
        
        post_atten_layer_norm_size = block_shapes["post_attention_layernorm.weight"][0]
        self.post_attention_layernorm = LayerNorm(post_atten_layer_norm_size, eps=config.layer_norm_epsilon)

        mlp_prefix = "mlp."
        mlp_shapes = {name.split(mlp_prefix)[-1]: shape for name, shape in block_shapes.items() if mlp_prefix in name}
        self.mlp = PrunedBloomMLP(config, mlp_shapes)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout
        
class PrunedBloomModel(BloomModel):
    def __init__(self, config: BloomConfig, state_dict_shapes, num_heads):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = num_heads

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Pruned Transformer blocks
        pruned_bloom_blocks = []
        for layer_num in range(config.num_hidden_layers):
            layer_prefix = f"transformer.h.{layer_num}."
            block_shapes = {name.split(layer_prefix)[-1]: shape for name, shape in state_dict_shapes.items() if layer_prefix in name}
            block = PrunedBloomBlock(config, block_shapes, num_heads[layer_num])
            pruned_bloom_blocks.append(block)
            
        self.h = nn.ModuleList(pruned_bloom_blocks)

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids = None,
        past_key_values = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        **deprecated_arguments,
    ):
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            print(i)
            num_heads = self.num_heads[i]
            alibi = build_alibi_tensor(attention_mask, num_heads, dtype=hidden_states.dtype)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
class PrunedBloomForCausalLM(BloomForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: BloomConfig, state_dict_shapes_path):
        super().__init__(config)
        state_dict_shapes, num_heads = pkl.load(open(state_dict_shapes_path, "rb"))
        
        self.transformer = PrunedBloomModel(config, state_dict_shapes, num_heads)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()