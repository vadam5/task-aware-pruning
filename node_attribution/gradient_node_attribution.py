import logging
import time
import torch
from collections import OrderedDict

from transformers import AutoTokenizer, BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
from node_attribution.bloom_for_gradient_node_attribution import BloomForCausalLMForNodeAttribution
from node_attribution.utils import count_params

# from bloom_for_gradient_node_attribution import BloomForCausalLMForNodeAttribution
# from utils import count_params


class BloomNodeAttributor:
    def __init__(self, model_size):
        self.load_bloom_model(model_size)

    def load_bloom_model(self, model_size):
        logging.info(f"Starting to load bigscience/bloom-{model_size} ...")
        self.model = self.load_split_qk_and_v_model(model_size)    
        self.tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-{model_size}")
        
        logging.info(f"bigscience/bloom-{model_size} loaded, counting params and calculating variables ...")
        
        self.total_params, self.base_params = count_params(self.model)
        self.num_blocks = self.model.transformer.config.num_hidden_layers
        self.num_heads = self.model.config.n_head
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.model_params = self.model.state_dict()
        
        print(f"Finished loading bigscience/bloom-{model_size}.\n\
                    Num Transformer Blocks: {self.num_blocks}\n\
                    Num Attention Heads: {self.num_heads}\n\
                    Head Dim: {self.head_dim}\n\
                    Hidden Size: {self.hidden_size}\n\
                    Base Model Param Count: {self.base_params:,}\n\
                    Total Param Count (w/ LM Head): {self.total_params:,}"
        )
        
    def load_split_qk_and_v_model(self, size):
        temp_model = BloomForCausalLM.from_pretrained(f"bigscience/bloom-{size}")
        temp_model_statedict = temp_model.state_dict()
        hidden_size = temp_model.config.hidden_size
        new_state_dict = OrderedDict()
        
        qk_index = torch.tensor([i for i in range(hidden_size * 2)])
        v_index = torch.tensor([i for i in range(hidden_size * 2, hidden_size * 3)])

        for layer_name in temp_model_statedict:
            if "query_key_value" in layer_name:
                qk_name = layer_name.replace("query_key_value", "query_key")
                v_name = layer_name.replace("query_key_value", "value")
                
                qkv_weights = temp_model_statedict[layer_name]
                
                qk_weights = torch.index_select(qkv_weights, 0, qk_index)
                new_state_dict[qk_name] = qk_weights
                
                if "bias" in layer_name:
                    prev_v_weights = layer_name.replace("query_key_value.bias", "value.weight")
                    new_state_dict.move_to_end(prev_v_weights)
                
                v_weights = torch.index_select(qkv_weights, 0, v_index)
                new_state_dict[v_name] = v_weights
                
                
            else:
                layer_value = temp_model_statedict[layer_name]
                new_state_dict[layer_name] = layer_value
                
        bloom_config = BloomConfig(
            vocab_size=250880,
            hidden_size=1024,
            n_layer=24,
            n_head=16,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            apply_residual_connection_post_layernorm=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pretraining_tp=1,  # TP rank used when training with megatron
            slow_but_exact=False,
            attention_softmax_in_fp32=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=True,
            offset_alibi=100,
            pad_token_id=3,
            seq_length=2048,
            skip_bias_add=True,
            skip_bias_add_qkv=False,
            unk_token_id=0,
            )
        
        temp_model = None
        model = BloomForCausalLMForNodeAttribution(bloom_config)
        model.load_state_dict(new_state_dict)
        
        return model

    def forward_pass(self, inputs, atten_mask):
        outputs = self.model(input_ids=inputs, attention_mask=atten_mask, return_dict=True)
        
        return outputs

    def calc_node_contributions(self, data):
        contributions = {}
        line_num = 1
        
        for line in data:
            print(line_num)
            line_num += 1
            inputs = self.tokenizer(line, return_tensors="pt")
            atten_mask = inputs['attention_mask']
            input_ids = inputs['input_ids']
            
            for i in range(len(input_ids[0])):
                token_id = input_ids[0][i].item()
                input_upto_index = torch.index_select(input_ids, 1, torch.tensor([j for j in range(i + 1)]))
                atten_mask_upto_index = torch.index_select(atten_mask, 1, torch.tensor([j for j in range(i + 1)]))
                outputs_upto_index = self.model(input_ids=input_upto_index, attention_mask=atten_mask_upto_index, return_dict=True)
                outputs_upto_index.logits[0][i][token_id].backward()
            
                for block_id in range(self.num_blocks):
                    dense_4h_to_h_gradients = torch.mean(self.model.transformer.h[block_id].mlp.dense_4h_to_h_activations.grad[0], 0)
                    mlp_param_name = f"transformer.h.{block_id}.mlp.dense_4h_to_h.weight"
                    
                    if mlp_param_name not in contributions:
                        contributions[mlp_param_name] = []
                    
                    contributions[mlp_param_name].append(dense_4h_to_h_gradients)
                    
                    value_gradients = torch.mean(self.model.transformer.h[block_id].self_attention.value_activations.grad[0], 0)
                    value_param_name = f"transformer.h.{block_id}.self_attention.value.weight"
                    
                    if value_param_name not in contributions:
                        contributions[value_param_name] = []
                        
                    contributions[value_param_name].append(value_gradients)
                    
                    # query_key_gradients = torch.mean(self.model.transformer.h[block_id].self_attention.query_key_activations.grad[0], 0)
                    # query_key_param_name = f"transformer.h.{block_id}.self_attention.query_key.weight"
                    
                    # if query_key_param_name not in contributions:
                    #     contributions[query_key_param_name] = []
                        
                    # contributions[query_key_param_name].append(query_key_gradients)
                       
        contributions = [(key, torch.stack(value)) for key, value in contributions.items()]
        return contributions
  
def get_attributions(model_size, data):
    attributor = BloomNodeAttributor(model_size)
    contributions = attributor.calc_node_contributions(data)
    
    avg_contribution = OrderedDict()
    max_contribution = OrderedDict()
    
    # Get average and max contributions for each node over the whole sequence
    for layer in contributions:
        layer_name, full_seq_contributions = layer
        avg = torch.mean(full_seq_contributions, dim=0)
        max = torch.amax(full_seq_contributions, dim=0).values
        
        avg_contribution[layer_name] = avg
        max_contribution[layer_name] = max
        
    return avg_contribution, max_contribution, attributor.model, attributor.model_params
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    model_size = "560m"
    data = ["Hello, I am an AlexPrize chatbot", "This is a second sentence"]
    get_attributions(model_size, data)