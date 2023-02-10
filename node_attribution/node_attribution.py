import logging
import time
import torch

from transformers import AutoTokenizer, BloomForCausalLM
from bloom_for_node_attribution import BloomForCausalLMForNodeAttribution
from utils import count_params


def load_bloom_model(model_size):
    logging.info(f"Starting to load bigscience/bloom-{model_size} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-{model_size}")
    model = BloomForCausalLMForNodeAttribution.from_pretrained(f"bigscience/bloom-{model_size}")
    
    logging.info(f"bigscience/bloom-{model_size} loaded, counting params and calculating variables ...")
    
    total_params, base_params = count_params(model)
    num_blocks = model.transformer.config.num_hidden_layers
    num_heads = model.config.n_head
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    model_params = model.state_dict()
    
    logging.info(f"Finished loading bigscience/bloom-{model_size}.\n\
                   Num Transformer Blocks: {num_blocks}\n\
                   Num Attention Heads: {num_heads}\n\
                   Head Dim: {head_dim}\n\
                   Hidden Size: {hidden_size}\n\
                   Base Model Param Count: {base_params:,}\n\
                   Total Param Count (w/ LM Head): {total_params:,}"
    )
                 
    return model, model_params, tokenizer, num_blocks, num_heads, head_dim
                 

def forward_pass(model, tokenizer, line):
    inputs = tokenizer(line, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        max_new_tokens=1, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        return_dict_in_generate=True
    )
    batch_size = len(inputs["input_ids"])
    seq_length = len(inputs["input_ids"][0])
    
    return outputs, batch_size, seq_length


def calc_node_contributions(model, model_params, tokenizer, num_blocks, num_heads, head_dim, data):
    contributions_to_data = []
    
    for line in data:
        contributions_to_line = []
        outputs, _, seq_length = forward_pass(model, tokenizer, line)
        model_activations = outputs.activations[0]
        sequence = outputs.sequences[0]
        
        # LM head
        head_contributions, weight_product_sum = lm_head_contributions(
            model_params=model_params, 
            model_activations=model_activations, 
            sequence=sequence,
        )
        contributions_to_line.extend(head_contributions)
        
        # Transformer blocks
        for block_id in range(num_blocks - 1, -1):
            block_id = str(block_id)
            block_contributions, weight_product_sum = transformer_block_contributions(
                model_params=model_params,
                model_activations=model_activations,
                weight_product_sum=weight_product_sum,
                block_id=block_id,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            contributions_to_line.extend(block_contributions)
        contributions_to_data.append(contributions_to_line)
        
    return contributions_to_data
            
        
def lm_head_contributions(model_params, model_activations, sequence):
    """Final hidden states to token logit contributions"""
    
    # weights have shape: (vocab_size x hidden size)
    lm_head_weights = model_params["lm_head.weight"]

    # Only care about weights connected to sequence tokens and none of the others
    seq_token_weights = torch.index_select(lm_head_weights, 0, sequence[:-1])
    hidden_state_activations = model_activations["transformer"]["ln_f"].squeeze()

    # Contribution of layer normed final hidden states to token logit
    contributions = torch.mul(seq_token_weights, hidden_state_activations)
    contributions = [("lm_head", contributions)]
    
    return contributions, seq_token_weights


def transformer_block_contributions(
    model_params, 
    model_activations, 
    weight_product_sum, 
    block_id, 
    seq_length, 
    num_heads, 
    head_dim
):
    contributions = []
    
    mlp_contributions, weight_product_sum = mlp_contributions(
        model_params=model_params,
        model_activations=model_activations,
        weight_product_sum=weight_product_sum,
        block_id=block_id,
        seq_length=seq_length,
    )
    contributions.extend(mlp_contributions)
    
    return contributions, weight_product_sum
    
    
def mlp_contributions(model_params, model_activations, weight_product_sum, block_id, seq_length):
    contributions = []
    
    # Last MLP layer input contribution to output, delta_x (activations), w_xy (weights)
    sub_block = "mlp"
    layer_name = "dense_h_to_4h"
    param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
    mlp_dense_4h_to_h_contributions, weight_product_sum = feed_forward_contributions(
        model_params=model_params, 
        model_activations=model_activations, 
        weight_product_sum=weight_product_sum, 
        block_id=block_id, 
        seq_length=seq_length, 
        sub_block=sub_block,
        layer_name=layer_name, 
        param_name=param_name,
    )
    contributions.append((param_name, mlp_dense_4h_to_h_contributions))
    
    # First MLP layer input contribution to output
    sub_block = "self_attention"
    layer_name = "dense"
    param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
    merged_head_to_4h_contributions, weight_product_sum = feed_forward_contributions(
        model_params=model_params, 
        model_activations=model_activations, 
        weight_product_sum=weight_product_sum, 
        block_id=block_id, 
        seq_length=seq_length, 
        sub_block=sub_block,
        layer_name=layer_name, 
        param_name=param_name,
    )
    contributions.append((param_name, merged_head_to_4h_contributions))
    
    return contributions, weight_product_sum
    
    
def feed_forward_contributions(
    model_params, 
    model_activations, 
    weight_product_sum, 
    block_id, 
    seq_length, 
    sub_block, 
    layer_name, 
    param_name
):
    # delta_x (activations), w_xy (weights)
    activations = model_activations["transformer"]["h"][block_id][sub_block][layer_name]
    weights = model_params[param_name]
    input_size = weights.shape[1]
    output_size = weights.shape[0]
    
    # Give weights shape (seq length x output size x input size)
    # In each (output size x input size) matrix, each row's elements are the weights from an 
    # input node to the output node corresponding to that rows index
    weights = weights.expand(seq_length, output_size, input_size)
    
    # w_yz
    weight_product_sum = weight_product_sum.unsqueeze(-1)
    weight_product_sum = weight_product_sum.expand(seq_length, output_size, input_size)
    
    # Multiply current layer weights with previous layer weights, w_xy * w_yz
    # Element-wise multiply each column by the output layer's weights to the final layer to get 
    # input layer's contribution to final prediction
    weight_product = torch.mul(weights, weight_product_sum)
    
    # Sum over column's elements (aka all weights from one input node to all output nodes) to 
    # have weight matrix for next layer
    # sum(w_xy * w_yz) over all y, used for next computation
    weight_product_sum = torch.sum(weight_product, 1)
    
    # Element-wise multiply each weight row by the input node's activation
    # Each column in contribution contains one input node's weights to every output node
    # w_xy * w_yz * delta_x and sum(w_xy * w_yz * delta_x) over all y
    contributions = torch.mul(weight_product_sum, activations)
    
    return contributions, weight_product_sum
 
  
def main(model_size, data):
    model, model_params, tokenizer, num_blocks, num_heads, head_dim = load_bloom_model(model_size)
    contributions_to_data = calc_node_contributions(
        model, 
        model_params, 
        tokenizer, 
        num_blocks, 
        num_heads, 
        head_dim, 
        data
    )
    
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    model_size = "560m"
    data = ["Hello, I am an AlexPrize chatbot"]
    main(model_size, data)