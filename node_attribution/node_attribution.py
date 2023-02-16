import logging
import time
import torch

from transformers import AutoTokenizer, BloomForCausalLM
from bloom_for_node_attribution import BloomForCausalLMForNodeAttribution
from utils import count_params

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
        
        logging.info(f"Finished loading bigscience/bloom-{model_size}.\n\
                    Num Transformer Blocks: {self.num_blocks}\n\
                    Num Attention Heads: {self.num_heads}\n\
                    Head Dim: {self.head_dim}\n\
                    Hidden Size: {self.hidden_size}\n\
                    Base Model Param Count: {self.base_params:,}\n\
                    Total Param Count (w/ LM Head): {self.total_params:,}"
        )

    def forward_pass(self, line):
        inputs = self.tokenizer(line, return_tensors="pt")
        outputs = self.model.generate(
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


    def calc_node_contributions(self, data):
        contributions = []
        
        for line in data:
            outputs, _, seq_length = self.forward_pass(line)
            sequence = outputs.sequences[0]
            calculate = SequenceContributionCalculator(
                model_params = self.model_params,
                model_activations = outputs.activations[0],
                sequence = outputs.sequences[0],
                seq_length = seq_length,
                num_heads = self.num_heads,
                head_dim = self.head_dim,
            )
            # LM head
            calculate.lm_head_contributions()
            
            # Transformer blocks
            for block_id in reversed(range(self.num_blocks)):
                block_id = str(block_id)
                calculate.transformer_block_contributions(block_id)
                
            contributions.append(calculate.contributions)
        
        return contributions
            
class SequenceContributionCalculator:
    def __init__(self, model_params, model_activations, sequence, seq_length, num_heads, head_dim):
        self.model_params = model_params
        self.model_activations = model_activations
        self.sequence = sequence
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.contributions = []
        self.prev_normalized_contributions = None
        self.batch_size = 1
        
    def update_prev_normalized_contributions(self, contributions):
        contributions = contributions.squeeze()
        contributions_dim = contributions.shape[-1]
        contributions_max = torch.max(contributions, -1).values
        contributions_min = torch.min(contributions, -1).values
        min_max_diff = torch.sub(contributions_max, contributions_min)
        
        expanded_contributions_min = contributions_min.unsqueeze(-1).expand(self.seq_length, contributions_dim)
        expanded_min_max_diff = min_max_diff.unsqueeze(-1).expand(self.seq_length, contributions_dim)

        contributions_minus_min = torch.sub(contributions, expanded_contributions_min)
        result = torch.div(contributions_minus_min, expanded_min_max_diff)
        
        return result
        
               
    def lm_head_contributions(self):
        """Final hidden states to token logit contributions"""
        
        layer_name = "lm_head"
        param_name = f"{layer_name}.weight"
        hidden_state_activations = self.model_activations[layer_name].squeeze()
        
        # weights have shape: (vocab_size x hidden size)
        lm_head_weights = self.model_params[param_name]

        # Only care about weights connected to sequence tokens and none of the others
        seq_token_weights = torch.index_select(lm_head_weights, 0, self.sequence[:-1])
    
        # Contribution of layer normed final hidden states to token logit
        contributions = torch.mul(seq_token_weights, hidden_state_activations)
        self.contributions.append((param_name, contributions))
        self.prev_normalized_contributions = self.update_prev_normalized_contributions(contributions)
        self.weight_product_sum = seq_token_weights
        
    def transformer_block_contributions(self, block_id):
        self.mlp_contributions(block_id)
        self.multihead_atten_contributions(block_id)
        
    def mlp_contributions(self, block_id):
        # Last MLP layer input contribution to output, delta_x (activations), w_xy (weights)
        sub_block = "mlp"
        layer_name = "dense_4h_to_h"
        param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
        mlp_dense_4h_to_h_contributions = self.feed_forward_contributions(
            block_id=block_id,
            sub_block=sub_block,
            layer_name=layer_name, 
            param_name=param_name,
        )
        self.contributions.append((param_name, mlp_dense_4h_to_h_contributions))
        
        # First MLP layer input contribution to output
        sub_block = "mlp"
        layer_name = "dense_h_to_4h"
        param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
        mlp_dense_h_to_4h_contributions = self.feed_forward_contributions( 
            block_id=block_id, 
            sub_block=sub_block,
            layer_name=layer_name, 
            param_name=param_name,
        )
        self.contributions.append((param_name, mlp_dense_h_to_4h_contributions))
        
    def feed_forward_contributions(self, block_id, sub_block, layer_name, param_name):
        # delta_x (activations), w_xy (weights)
        activations = self.model_activations["transformer"]["h"][block_id][sub_block][layer_name]
        weights = self.model_params[param_name]
        
        input_size = weights.shape[1]
        output_size = weights.shape[0]
        
        # Give weights shape (seq length x output size x input size)
        # In each (output size x input size) matrix, each row's elements are the weights from an 
        # input node to the output node corresponding to that rows index
        weights = weights.expand(self.seq_length, output_size, input_size)
        
        # w_yz
        prev_contributions = self.prev_normalized_contributions.unsqueeze(-1)
        prev_contributions = prev_contributions.expand(self.seq_length, output_size, input_size)
        
        # Multiply current layer weights with previous layer weights, w_xy * w_yz
        # Element-wise multiply each column by the output layer's weights to the final layer to get 
        # input layer's contribution to final prediction
        weight_product = torch.mul(weights, prev_contributions)
        
        # Sum over column's elements (aka all weights from one input node to all output nodes) to 
        # have weight matrix for next layer
        # sum(w_xy * w_yz) over all y, used for next computation
        weight_product_sum = torch.sum(weight_product, 1)
        
        # Element-wise multiply each weight row by the input node's activation
        # Each column in contribution contains one input node's weights to every output node
        # w_xy * w_yz * delta_x and sum(w_xy * w_yz * delta_x) over all y
        contributions = torch.mul(weight_product_sum, activations)
        self.prev_normalized_contributions = self.update_prev_normalized_contributions(contributions)
        
        return contributions
    
    def multihead_atten_contributions(self, block_id):
        # Merged multi-head attention contribution to dense layer outputs
        sub_block = "self_attention"
        layer_name = "dense"
        param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
        merged_atten_head_contributions = self.feed_forward_contributions(block_id, sub_block, layer_name, param_name)
        self.contributions.append((param_name, merged_atten_head_contributions))
        
        value_contributions = self.value_layer_contributions(block_id)
        query_key_attn_contribution = self.query_key_atten_contributions(block_id)
        query_contributions = self.query_layer_contributions(block_id, query_key_attn_contribution)
        key_contributions = self.key_layer_contributions(block_id, query_key_attn_contribution)
        
        # Now arrange and combine the key, query, and value weight product sums and contributions
        # to form the weight product sum and contributions for the fused qkv layer output
        # for shape, see https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/models/bloom/modeling_bloom.py#L297
        query_contributions = query_contributions.unsqueeze(0).transpose(1, 2).unsqueeze(-2)
        key_contributions = key_contributions.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(-2)
        value_contributions = value_contributions.unsqueeze(0).transpose(1, 2).unsqueeze(-2)
        
        # fuse'em
        fused_qkv_contributions = torch.cat([query_contributions, key_contributions, value_contributions], -2)
        fused_qkv_contributions = fused_qkv_contributions.view(self.batch_size, self.seq_length, self.num_heads * 3 * self.head_dim)
        
        param_name = f"transformer.h.{block_id}.self_attention.query_key_value_fused_output.weight"
        self.contributions.append((param_name, fused_qkv_contributions))
        self.prev_normalized_contributions = self.update_prev_normalized_contributions(fused_qkv_contributions)
        
        sub_block = "self_attention"
        layer_name = "query_key_value"
        param_name = f"transformer.h.{block_id}.{sub_block}.{layer_name}.weight"
        block_input_contributions = self.feed_forward_contributions(block_id, sub_block, layer_name, param_name)
        self.contributions.append((param_name, block_input_contributions))
        
    
    def value_layer_contributions(self, block_id):
        """Value layer contributions to merged head output"""
        
        # Value activations are the weights for the query_key output
        value_layer_activations = self.model_activations["transformer"]["h"][block_id]["self_attention"]["value_layer"]

        # Query key output are the weights for the value activations
        attention_probs = torch.clone(self.model_activations["transformer"]["h"][block_id]["self_attention"]["attention_probs_reshaped"])

        # Need to reshape merged head contribution to multiply with attention_probs 
        # and value (layer num heads x seq_length x head dim)
        prev_contributions = torch.clone(self.prev_normalized_contributions).view(self.seq_length, self.num_heads, self.head_dim)
        prev_contributions = prev_contributions.transpose(0, 1)

        # Need to add an extra dim to the weight product sum because each column 
        # of the attention prob needs to be multiplied by the whole weight product sum
        prev_contributions = prev_contributions.unsqueeze(1)
        prev_contributions = prev_contributions.expand(self.num_heads, self.seq_length, self.seq_length, self.head_dim)

        # Also need to expand attention probs to have final dim=head dim for elementwise multiplication
        expanded_attention_probs = attention_probs.transpose(1, 2)
        expanded_attention_probs = expanded_attention_probs.unsqueeze(-1)
        expanded_attention_probs = expanded_attention_probs.expand(self.num_heads, self.seq_length, self.seq_length, self.head_dim)

        # softmax(query x key) are the weights for the value layer
        # Multiply softmax(query x key) output by next layer's weight product sum to 
        # get the weight product some for value layer contribution
        # Need to elementwise multiply (16, 10, 10) each column of this matrix 
        # by each column of the current weight product sum matrix.
        value_weight_product = torch.mul(prev_contributions, expanded_attention_probs)

        # Now we can sum over that extra dim we have. Each column in the most inner 
        # matrix represents one number in the value layer's weight product sum contribution.
        value_weight_product_sum = torch.sum(value_weight_product, 2)

        # If I element wise multiplied the value layer and this current weight product sum, 
        # it gives the value layer output's contribution to the final prediction.
        value_contributions = torch.mul(value_layer_activations, value_weight_product_sum)

        return value_contributions
    
    
    def query_key_atten_contributions(self, block_id):
        # Value activations are the weights for the query_key atten output
        value_layer_activations = torch.clone(self.model_activations["transformer"]["h"][block_id]["self_attention"]["value_layer"])
        
        # Activations need for the previous contributions value
        query_key_attn_probs = self.model_activations["transformer"]["h"][block_id]["self_attention"]["attention_probs"]

        # Same reshaping that was needed for the value contribution calculation
        query_key_attn_weight_product_sum = torch.clone(self.prev_normalized_contributions).view(self.seq_length, self.num_heads, self.head_dim)
        query_key_attn_weight_product_sum = query_key_attn_weight_product_sum.transpose(0, 1)

        # Need to multiply each row in the product sum by every row in the value weight matrix, 
        # so replicating each product sum row ((value weight matrix num rows) = seq_length) amount of times
        # Note I am expanding a different dimension here compared to what I did for the value weight product sum 
        query_key_attn_weight_product_sum = query_key_attn_weight_product_sum.unsqueeze(2)
        query_key_attn_weight_product_sum = query_key_attn_weight_product_sum.expand(self.num_heads, self.seq_length, self.seq_length, self.head_dim)

        # Also need to expand the value weight matrix for this elementwise multiplication
        expanded_value_layer_activations = value_layer_activations.unsqueeze(1)
        expanded_value_layer_activations = expanded_value_layer_activations.expand(self.num_heads, self.seq_length, self.seq_length, self.head_dim)

        # For each row in the weight product sum, multiply it by every row in the value weight matrix
        query_key_attn_weight_product = torch.mul(query_key_attn_weight_product_sum, expanded_value_layer_activations)

        # Sum over the head_dim dimension, which in this case, can be thought of as the the output dimension
        query_key_attn_weight_product_sum = torch.sum(query_key_attn_weight_product, -1)
        
        # Multiplying by the query key activations gives the query key contribution. Don't actually need these contriution values though.
        query_key_attn_contribution = torch.mul(query_key_attn_probs.squeeze(), query_key_attn_weight_product_sum)
        query_key_attn_contribution = query_key_attn_contribution.transpose(0, 1).reshape(self.seq_length, self.seq_length * self.num_heads)
        query_key_attn_contribution = self.update_prev_normalized_contributions(query_key_attn_contribution)
        query_key_attn_contribution = query_key_attn_contribution.reshape(self.seq_length, self.num_heads, self.seq_length).transpose(0, 1)
        
        return query_key_attn_contribution
    
    
    def query_layer_contributions(self, block_id, query_key_attn_contribution):
        """Query contributions to query_key attention weights"""
        
        # The key activations are the query's weights when calcualting the query contibution
        key_activations = torch.clone(self.model_activations["transformer"]["h"][block_id]["self_attention"]["key_layer"])
        query_activations = self.model_activations["transformer"]["h"][block_id]["self_attention"]["query_layer"]
    
        # Treat this weight product sum like I treated the weight product sum in query_key_attn_weight_product_sum
        query_weight_product_sum = torch.clone(query_key_attn_contribution).unsqueeze(2)
        query_weight_product_sum = query_weight_product_sum.expand(self.num_heads, self.seq_length, self.head_dim, self.seq_length)

        # Treat the key activations like I treated the value activations when calculating the query_key_attn_weight_product_sum
        expanded_key_layer_activations = key_activations.unsqueeze(1)
        expanded_key_layer_activations = expanded_key_layer_activations.expand(self.num_heads, self.seq_length, self.head_dim, self.seq_length)

        # For each row in the weight product sum, multiply it by every row in the key weight matrix
        query_weight_product = torch.mul(query_weight_product_sum, expanded_key_layer_activations)
        query_weight_product_sum = torch.sum(query_weight_product, -1)

        # Treat the query activations like I treated the query_key_attn activations
        query_contributions = torch.mul(query_activations, query_weight_product_sum)
        
        return query_contributions
    
    
    def key_layer_contributions(self, block_id, query_key_attn_contribution):
        """Key contributions to the query_key attention weights"""
        
        # The query activations are the key's weights when calcualting the key contibution
        query_activations = torch.clone(self.model_activations["transformer"]["h"][block_id]["self_attention"]["query_layer"])
        key_activations = self.model_activations["transformer"]["h"][block_id]["self_attention"]["key_layer"]
        
        # Treat key matrix like value matrix when calculating value contribution
        key_weight_product_sum = torch.clone(query_key_attn_contribution).unsqueeze(1)
        key_weight_product_sum = key_weight_product_sum.expand(self.num_heads, self.head_dim, self.seq_length, self.seq_length)

        expanded_query_activations = query_activations.transpose(1, 2)
        expanded_query_activations = expanded_query_activations.unsqueeze(-1)
        expanded_query_activations = expanded_query_activations.expand(self.num_heads, self.head_dim, self.seq_length, self.seq_length)

        key_weight_product = torch.mul(key_weight_product_sum, expanded_query_activations)
        key_weight_product_sum = torch.sum(key_weight_product, 2)
        key_contributions = torch.mul(key_activations, key_weight_product_sum)
        
        return key_contributions
  
def main(model_size, data):
    attributor = NodeAttributor(model_size)
    contributions = attributor.calc_node_contributions(data)[0]
    
    avg_contribution = {}
    max_contribution = {}
    
    # Get average and max contributions for each node over the whole sequence
    for layer in contributions:
        layer_name, full_seq_contributions = layer
        avg = torch.mean(full_seq_contributions.squeeze(), 0)
        max = torch.max(full_seq_contributions.squeeze(), 0).values
        
        avg_contribution[layer_name] = avg
        max_contribution[layer_name] = max
        
        print(layer_name, max.shape, max)
        
    return avg_contribution, max_contribution
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    model_size = "560m"
    data = ["Hello, I am an AlexPrize chatbot"]
    main(model_size, data)