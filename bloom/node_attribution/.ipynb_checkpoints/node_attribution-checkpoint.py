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
    
    logging.info(f"Finished loading bigscience/bloom-{model_size}.\n\
                   Num Transformer Blocks: {num_blocks}\n\
                   Num Attention Heads: {num_heads}\n\
                   Head Dim: {head_dim}\n\
                   Hidden Size: {hidden_size}\n\
                   Base Model Param Count: {base_params:,}\n\
                   Total Param Count (w/ LM Head): {total_params:,}"
    )
                 
    return model, tokenizer, num_blocks, num_heads, head_dim
                 

def forward_pass(model, tokenizer, line):
    inputs = tokenizer(line, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=1, do_sample=True, top_k=50, top_p=0.95, return_dict_in_generate=True)
    batch_size = len(inputs["input_ids"])
    seq_length = len(inputs["input_ids"][0])
    
    return outputs, batch_size, seq_length


def calc_node_contributions(model, tokenizer, num_blocks, num_heads, head_dim, data):
    node_corpus_contributions = []
    
    for line in data:
        node_line_contributions = []
        outputs, batch_size, seq_length = forward_pass(model, tokenizer, line)
        
        # LM head
        contributions, weight_product_sum = lm_head_contributions(model, outputs)
        node_line_contributions.append({"lm_head_contributions": contributions})
        
        # Transformer blocks
        # for block_id in range(num_blocks - 1, -1):
        
        node_corpus_contributions.append(node_line_contributions)
        
    return node_corpus_contributions
            
        
def lm_head_contributions(model, outputs):
    """Final hidden states to token logit contributions"""
    
    # weights have shape: (vocab_size x hidden size)
    lm_head_weights = next(param for param in model.lm_head.parameters())

    # Only care about weights connected to sequence tokens and none of the others
    seq_token_weights = torch.index_select(lm_head_weights, 0, outputs.sequences[0][:-1])
    hidden_state_activations = outputs.activations[0]["transformer"]["ln_f"].squeeze()

    # Contribution of layer normed final hidden states to token logit
    contributions = torch.mul(seq_token_weights, hidden_state_activations)
    
    return contributions, seq_token_weights

        
def main(model_size, data):
    model, tokenizer, num_blocks, num_heads, head_dim = load_bloom_model(model_size)
    node_contributions = calc_node_contributions(model, tokenizer, num_blocks, num_heads, head_dim, data)
    
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    model_size = "560m"
    data = ["Hello, I am an AlexPrize chatbot"]
    main(model_size, data)