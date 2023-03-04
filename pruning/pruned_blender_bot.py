import math
import pickle as pkl

import torch
import torch.nn as nn

from transformers.models.blenderbot.modeling_blenderbot import (
    BlenderbotAttention, 
    BlenderbotDecoder,
    BlenderbotDecoderLayer,
    BlenderbotEncoder,
    BlenderbotEncoderLayer, 
    BlenderbotForConditionalGeneration,
    BlenderbotLearnedPositionalEmbedding,
    BlenderbotModel
)

from transformers.activations import ACT2FN
        
class PrunedBlenderbotEncoderLayer(BlenderbotEncoderLayer):
    def __init__(self, config, layer_shapes):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        encoder_ffn_dim = layer_shapes["fc1.weight"][0]
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
class PrunedBlenderbotDecoderLayer(BlenderbotDecoderLayer):
    def __init__(self, config, layer_shapes):
        super().__init__(config)
        self.embed_dim = config.d_model

        self.self_attn = BlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BlenderbotAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        decoder_ffn_dim = layer_shapes["fc1.weight"][0]
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
class PrunedBlenderbotEncoder(BlenderbotEncoder):
    def __init__(self, config, state_dict_shapes, embed_tokens = None):
        super().__init__(config, embed_tokens)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        pruned_layers = []
        for layer_id in range(config.encoder_layers):
            layer_prefix = f"model.encoder.layers.{layer_id}."
            layer_shapes = {name.split(layer_prefix)[-1]: shape for name, shape in state_dict_shapes.items() if layer_prefix in name}
            layer = PrunedBlenderbotEncoderLayer(config, layer_shapes)
            pruned_layers.append(layer)
            
        self.layers = nn.ModuleList(pruned_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
class PrunedBlenderbotDecoder(BlenderbotDecoder):
    def __init__(self, config, state_dict_shapes, embed_tokens = None):
        super().__init__(config, embed_tokens)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        pruned_layers = []
        for layer_id in range(config.decoder_layers):
            layer_prefix = f"model.decoder.layers.{layer_id}."
            layer_shapes = {name.split(layer_prefix)[-1]: shape for name, shape in state_dict_shapes.items() if layer_prefix in name}
            layer = PrunedBlenderbotDecoderLayer(config, layer_shapes)
            pruned_layers.append(layer)
            
        self.layers = nn.ModuleList(pruned_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
class PrunedBlenderbotModelForNodeAttribution(BlenderbotModel):
    _keys_to_ignore_on_load_missing = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config, state_dict_shapes):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PrunedBlenderbotEncoder(config, state_dict_shapes, self.shared)
        self.decoder = PrunedBlenderbotDecoder(config, state_dict_shapes, self.shared)

        # Initialize weights and apply final processing
        self.post_init()
        

class PrunedBlenderbotForConditionalGeneration(BlenderbotForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
    ]

    def __init__(self, config, state_dict_shapes_path):
        super().__init__(config)
        state_dict_shapes = pkl.load(open(state_dict_shapes_path, "rb"))
        
        self.model = PrunedBlenderbotModelForNodeAttribution(config, state_dict_shapes)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()