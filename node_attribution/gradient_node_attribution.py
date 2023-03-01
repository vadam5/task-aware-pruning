import logging
import time
import torch
from collections import OrderedDict

from transformers import AutoTokenizer
from node_attribution.bloom_for_gradient_node_attribution import BloomForCausalLMForNodeAttribution
from node_attribution.utils import count_params

# from bloom_for_gradient_node_attribution import BloomForCausalLMForNodeAttribution
# from utils import count_params


class NodeAttributor:
    def __init__(self, model_size):
        self.load_bloom_model(model_size)

    def load_bloom_model(self, model_size):
        logging.info(f"Starting to load bigscience/bloom-{model_size} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-{model_size}")
        self.model = BloomForCausalLMForNodeAttribution.from_pretrained(f"bigscience/bloom-{model_size}")
        
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

    def forward_pass(self, inputs, atten_mask):
        outputs = self.model(input_ids=inputs, attention_mask=atten_mask, return_dict=True)
        
        return outputs

    def calc_node_contributions(self, data):
        contributions = []
        
        for line in data:
            contributions_to_line = {}
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
                    
                    if mlp_param_name not in contributions_to_line:
                        contributions_to_line[mlp_param_name] = []
                    
                    contributions_to_line[mlp_param_name].append(dense_4h_to_h_gradients)
                    
                    weighted_value_layer_gradients = torch.mean(self.model.transformer.h[block_id].self_attention.dense_activations.grad[0], 0)
                    # tiled_weighted_value_layer_gradients = torch.tile(weighted_value_layer_gradients, (3,))
                    qkv_param_name = f"transformer.h.{block_id}.self_attention.value_layer.weight"
                    
                    if qkv_param_name not in contributions_to_line:
                        contributions_to_line[qkv_param_name] = []
                    
                    contributions_to_line[qkv_param_name].append(weighted_value_layer_gradients)
                    
                    # print(self.model.transformer.h[block_id].self_attention.query_key_value_activations.grad)
                    # query_key_value_output_gradients = self.model.transformer.h[block_id].self_attention.query_key_value_output_activations.grad[0]
                    # print(query_key_value_output_gradients.shape)
                    # qkv_param_name = f"transformer.h.{block_id}.self_attention.query_key_value_fused_output.weight"
                    # contributions_to_line.append((qkv_param_name, query_key_value_output_gradients))
                       
            contributions_to_line = [(key, torch.stack(value)) for key, value in contributions_to_line.items()]
            contributions.append(contributions_to_line)
                
        return contributions
  
def get_attributions(model_size, data):
    attributor = NodeAttributor(model_size)
    contributions = attributor.calc_node_contributions(data)[0]
    
    avg_contribution = OrderedDict()
    max_contribution = OrderedDict()
    final_contribution = OrderedDict()
    
    # Get average and max contributions for each node over the whole sequence
    for layer in contributions:
        layer_name, full_seq_contributions = layer
        avg = torch.mean(full_seq_contributions.squeeze(), 0)
        max = torch.max(full_seq_contributions.squeeze(), 0).values
        
        avg_contribution[layer_name] = avg
        max_contribution[layer_name] = max
        
    return avg_contribution, max_contribution, attributor.model, attributor.model_params
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    model_size = "560m"
    data = ["Hello, I am an AlexPrize chatbot"]
    get_attributions(model_size, data)