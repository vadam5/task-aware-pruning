import logging
import time
import torch
from collections import OrderedDict

from transformers import AutoTokenizer
from node_attribution.blender_bot_for_gradient_node_attribution import BlenderbotConditionalGenerationForNodeAttribution
from node_attribution.utils import count_params

# from blender_bot_for_gradient_node_attribution import BlenderbotConditionalGenerationForNodeAttribution
# from utils import count_params


class BlenderBotNodeAttributor:
    def __init__(self, model_size):
        self.load_blender_bot_model(model_size)

    def load_blender_bot_model(self, model_size):
        logging.info(f"Starting to load facebook/blenderbot-{model_size} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/blenderbot-{model_size}")
        self.model = BlenderbotConditionalGenerationForNodeAttribution.from_pretrained(f"facebook/blenderbot-{model_size}")
        
        logging.info(f"facebook/blenderbot-{model_size} loaded, counting params and calculating variables ...")
        
        self.total_params, self.base_params = count_params(self.model)
        self.num_encoder_layers = self.model.config.encoder_layers
        self.num_decoder_layers = self.model.config.decoder_layers
        self.num_encoder_heads = self.model.config.encoder_attention_heads
        self.num_decoder_heads = self.model.config.decoder_attention_heads
        self.hidden_size = self.model.config.d_model
        self.encoder_head_dim = self.hidden_size // self.num_encoder_heads
        self.decoder_head_dim = self.hidden_size // self.num_decoder_heads
        self.model_params = self.model.state_dict()
        
        print(f"Finished loading facebook/blenderbot-{model_size}.\n\
                    Num Encoder Transformer Blocks: {self.num_encoder_layers}\n\
                    Num Decoder Transformer Blocks: {self.num_decoder_layers}\n\
                    Num Encoder Attention Heads: {self.num_encoder_heads}\n\
                    Num Decoder Attention Heads: {self.num_decoder_heads}\n\
                    Encoder Head Dim: {self.encoder_head_dim}\n\
                    Decoder Head Dim: {self.decoder_head_dim}\n\
                    Hidden Size: {self.hidden_size}\n\
                    Base Model Param Count: {self.base_params:,}\n\
                    Total Param Count (w/ LM Head): {self.total_params:,}"
        )
        
        for name, weight in self.model_params.items():
            print(name, weight.shape)

    def forward_pass(self, inputs, atten_mask):
        outputs = self.model(input_ids=inputs, attention_mask=atten_mask, return_dict=True)
        
        return outputs

    def calc_node_contributions(self, data):
        contributions = {}
        line_num = 1
        
        for line in data:
            print(line_num)
            line_num += 1
            
            if line.startswith("chatbot"):
                input_seq, output_seq = line.split("user:")
                input_seq = input_seq.split("chatbot:")[-1].strip()
                output_seq = output_seq.strip()
            else:
                input_seq, output_seq = line.split("chatbot:")
                input_seq = input_seq.split("user:")[-1].strip()
                output_seq = output_seq.strip()
            
            inputs = self.tokenizer(input_seq, return_tensors="pt")
            input_ids = inputs.input_ids
            
            decoder_inputs = self.tokenizer(output_seq.strip(), return_tensors="pt")
            decoder_input_ids = decoder_inputs.input_ids
            # print(output_seq.strip())
            
            for i in range(len(decoder_input_ids[0])):
                token_id = decoder_input_ids[0][i].item()
                decoder_input_upto_index = torch.index_select(decoder_input_ids, 1, torch.tensor([j for j in range(i + 1)]))
                outputs_upto_index = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_upto_index)
                outputs_upto_index.logits[0][i][token_id].backward()
            
                # Encoder Layers
                for layer_id in range(self.num_encoder_layers):
                    dense_4h_to_h_gradients = torch.mean(self.model.model.encoder.layers[layer_id].fc1_activations.grad[0], 0)
                    mlp_param_name = f"model.encoder.layers.{layer_id}.fc2.weight"
                    
                    if mlp_param_name not in contributions:
                        contributions[mlp_param_name] = []
                    
                    contributions[mlp_param_name].append(dense_4h_to_h_gradients)
                    
                    query_layer_gradients = torch.mean(self.model.model.encoder.layers[layer_id].self_attn.q_proj_activations.grad[0], 0)
                    query_param_name = f"model.encoder.layers.{layer_id}.self_attn.q_proj.weight"
                    
                    if query_param_name not in contributions:
                        contributions[query_param_name] = []
                    
                    contributions[query_param_name].append(query_layer_gradients)
                    
                    key_layer_gradients = torch.mean(self.model.model.encoder.layers[layer_id].self_attn.k_proj_activations.grad[0], 0)
                    key_param_name = f"model.encoder.layers.{layer_id}.self_attn.k_proj.weight"
                    
                    if key_param_name not in contributions:
                        contributions[key_param_name] = []
                    
                    contributions[key_param_name].append(key_layer_gradients)
                    
                    value_layer_gradients = torch.mean(self.model.model.encoder.layers[layer_id].self_attn.v_proj_activations.grad[0], 0)
                    value_param_name = f"model.encoder.layers.{layer_id}.self_attn.v_proj.weight"
                    
                    if value_param_name not in contributions:
                        contributions[value_param_name] = []
                    
                    contributions[value_param_name].append(value_layer_gradients)
                    
                # Decoder Layers
                for layer_id in range(self.num_decoder_layers):
                    dense_4h_to_h_gradients = torch.mean(self.model.model.decoder.layers[layer_id].fc1_activations.grad[0], 0)
                    mlp_param_name = f"model.decoder.layers.{layer_id}.fc2.weight"
                    
                    if mlp_param_name not in contributions:
                        contributions[mlp_param_name] = []
                    
                    contributions[mlp_param_name].append(dense_4h_to_h_gradients)
                    
                    query_layer_gradients = torch.mean(self.model.model.decoder.layers[layer_id].self_attn.q_proj_activations.grad[0], 0)
                    query_param_name = f"model.decoder.layers.{layer_id}.self_attn.q_proj.weight"
                    
                    if query_param_name not in contributions:
                        contributions[query_param_name] = []
                    
                    contributions[query_param_name].append(query_layer_gradients)
                    
                    key_layer_gradients = torch.mean(self.model.model.decoder.layers[layer_id].self_attn.k_proj_activations.grad[0], 0)
                    key_param_name = f"model.decoder.layers.{layer_id}.self_attn.k_proj.weight"
                    
                    if key_param_name not in contributions:
                        contributions[key_param_name] = []
                    
                    contributions[key_param_name].append(key_layer_gradients)
                    
                    value_layer_gradients = torch.mean(self.model.model.decoder.layers[layer_id].self_attn.v_proj_activations.grad[0], 0)
                    value_param_name = f"model.decoder.layers.{layer_id}.self_attn.v_proj.weight"
                    
                    if value_param_name not in contributions:
                        contributions[value_param_name] = []
                    
                    contributions[value_param_name].append(value_layer_gradients)
                
                       
        contributions = [(key, torch.stack(value)) for key, value in contributions.items()]
        return contributions
  
def get_attributions(model_size, data):
    attributor = BlenderBotNodeAttributor(model_size)
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
    
    #model_size = "3B"
    model_size = "400M-distill"
    data = ["User: Hi my name is megan\nchatbot: Hello, I am an AlexPrize chatbot", "User: I like dogs\nchatbot: Hello, I am an AlexPrize chatbot"]
    get_attributions(model_size, data)