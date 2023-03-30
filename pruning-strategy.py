import torch
import numpy as np
import pickle as pkl

from node_attribution.gradient_node_attribution import get_attributions
from transformers import AutoTokenizer
from node_attribution.bloom_for_gradient_node_attribution import BloomForCausalLMForNodeAttribution
from node_attribution.utils import count_params

human_filtered_pairs = pkl.load(open("../data/44_human_filtered_conv_pairs.pkl", "rb"))
calibration_data = human_filtered_pairs[:22]

model_size = "1b1"
tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-{model_size}")
model = BloomForCausalLMForNodeAttribution.from_pretrained(f"bigscience/bloom-{model_size}")

print(f"bigscience/bloom-{model_size} loaded, counting params and calculating variables ...")

total_params, base_params = count_params(model)
num_blocks = model.transformer.config.num_hidden_layers
num_heads = model.config.n_head
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads
model_params = model.state_dict()

print(f"Finished loading bigscience/bloom-{model_size}.\n\
            Num Transformer Blocks: {num_blocks}\n\
            Num Attention Heads: {num_heads}\n\
            Head Dim: {head_dim}\n\
            Hidden Size: {hidden_size}\n\
            Base Model Param Count: {base_params:,}\n\
            Total Param Count (w/ LM Head): {total_params:,}"
        )


avg_contributions = pkl.load(open("../data/avg_contri_1b7_22pair_calibration.pkl", "rb"))

prune_percent = 0.40
num_params_to_prune = base_params * prune_percent
save_name = f"pruned_40percent_{model_size}_bloom.pt"
state_dict_save_name = f"pruned_40percent_{model_size}_bloom_state_dict_shapes.pkl"

index = 0
params_to_index = {}
for param_name in model_params.keys():
    if "bias" not in param_name:
        params_to_index[param_name] = index
        index += 1

index_to_params = {params_to_index[param_name]: param_name for param_name in params_to_index.keys()}

layer_shape_map = {}
layer_names, contribution_tensors = zip(*avg_contributions.items())
num_layers = len(layer_names)

for i in range(num_layers):
    layer_name = layer_names[i]
    layer_size = contribution_tensors[i].shape[0]

    if i != 0:
        next_layer_size = contribution_tensors[i - 1].shape[0]
    else:
        next_layer_size = 250880


    if i != (num_layers - 1):
        prev_layer_size = contribution_tensors[i + 1].shape[0]
    else:
        prev_layer_size = 1024

    layer_shape_map[layer_name] = {
        "prev_layer_size": prev_layer_size,
        "next_layer_size": next_layer_size,
        "current_layer_size": layer_size
    }

all_nodes = []

for layer_name, contribution_tensor in avg_contributions.items():
    for node_id, node in enumerate(contribution_tensor.tolist()):
        node_name = f"{layer_name}.{node_id}"
        all_nodes.append((node_name, node))
    
# Sort all the nodes
all_nodes.sort(key = lambda x:x[1])

all_layers = []

# Figure out which layers have the most to prune
for layer_name, contribution_tensor in avg_contributions.items():

    # Don't prune self_attention.dense.weight directly, use value matrix to decide what to prune
    if "self_attention.dense.weight" in layer_name:
        continue

    if "mlp.dense_h_to_4h.weight" in layer_name:
        continue

    if "self_attention.query_key_value.weight" in layer_name:
        continue

    if "transformer.ln_f.weight" in layer_name:
        continue

    if "lm_head.weight" in layer_name:
        continue

    # Get average contribution over the whole layer
    mean_contribution = torch.mean(contribution_tensor, 0).item()
    all_layers.append((layer_name, mean_contribution))

all_layers.sort(key = lambda x:x[1])

layer_masks = {}
num_params_pruned = 0
node_num = 0
min_nodes = 24
value_dim = 2

while num_params_pruned < num_params_to_prune:
    lowest_contr_layer_name = all_layers[0][0]
    shapes = layer_shape_map[lowest_contr_layer_name]
    stop_pruning_layer = False

    # Prune one node at time
    if lowest_contr_layer_name not in layer_masks:
        layer_contributions = avg_contributions[lowest_contr_layer_name]
        mask = torch.zeros_like(layer_contributions)
        sorted_contributions = torch.argsort(layer_contributions)
        num_pruned = 0

    else:
        mask = layer_masks[lowest_contr_layer_name][0]
        sorted_contributions = layer_masks[lowest_contr_layer_name][1]
        num_pruned = layer_masks[lowest_contr_layer_name][2]

    index_to_mask = sorted_contributions[num_pruned]
    mask[index_to_mask] = 1

    nodes_left = torch.numel(mask) - int(torch.sum(mask).item())

    # Keep from deleting all nodes in a layer
    if nodes_left > min_nodes:
        layer_masks[lowest_contr_layer_name] = (mask, sorted_contributions, num_pruned + 1)
        # num_params_pruned += shapes["prev_layer_size"]
        # num_params_pruned += shapes["next_layer_size"]
        num_params_pruned += hidden_size * 2
        node_num += 1
    else:
        stop_pruning_layer = True

    # Apply mask and update the layer mean in "all_layers"
    if stop_pruning_layer:
        new_layer_contr_score = float('inf')
    else:
        mean_array = np.ma.array(data=avg_contributions[lowest_contr_layer_name], mask=mask)
        new_layer_contr_score = mean_array.mean()

    # print(all_layers[0])
    all_layers[0] = (lowest_contr_layer_name, new_layer_contr_score)
    # print(all_layers[0])
    # print(f"Num params removed: {num_params_pruned}")
    # print(f"Num Nodes removed: {node_num}")
    # print("=====")

    # re-sort layers now that this one has been pruned and pick the lowest contributing layer again
    all_layers.sort(key = lambda x:x[1])


# Line up weights to prune and weights in the state dict
mask_index = 0
sorted_weight_index = 1
pruned_model_params = model_params.copy()

for layer_name in layer_masks.keys():
    if layer_name == "transformer.h.0.self_attention.query_key_value.weight":
        continue
    elif "self_attention.value_layer.weight" in layer_name:
        # Prune as input
        # Look at value matrix to decide what should be droped in "self_attention.dense.weight"
        value_reshape_mask = layer_masks[layer_name][mask_index].transpose(0, 1)[-1].reshape(num_heads * head_dim)
        num_nodes_to_drop = int(sum(value_reshape_mask).item())
        value_indices = layer_masks[layer_name][sorted_weight_index].transpose(0, 1)[-1].reshape(num_heads * head_dim)
        value_keep_index = torch.sort(value_indices[num_nodes_to_drop:]).values

        dense_layer_name = layer_name.replace("value_layer", "dense")
        pruned_input_weights = torch.index_select(pruned_model_params[dense_layer_name], -1, value_keep_index)
        pruned_model_params[dense_layer_name] = pruned_input_weights

        # Re-arrange mask to flatten shape
        reshaped_mask = layer_masks[layer_name][mask_index].view(num_heads * 3 * head_dim)
        rehsaped_indices = layer_masks[layer_name][sorted_weight_index].view(num_heads * 3 * head_dim)
        num_nodes_to_drop = int(sum(reshaped_mask).item())
        keep_index = torch.sort(rehsaped_indices[num_nodes_to_drop:]).values

        # Prune as output
        prev_layer_name = layer_name.replace("value_layer", "query_key_value")
        pruned_output_weights = torch.index_select(pruned_model_params[prev_layer_name], 0, keep_index)
        pruned_model_params[prev_layer_name] = pruned_output_weights

        # Also do bias term
        bias_layer_name = prev_layer_name.replace("weight", "bias")
        pruned_bias_weights = torch.index_select(pruned_model_params[bias_layer_name], 0, keep_index)
        pruned_model_params[bias_layer_name] = pruned_bias_weights

    else:
        # Prune when nodes are the input
        num_nodes_to_drop = int(sum(layer_masks[layer_name][mask_index]).item())
        keep_index = torch.sort(layer_masks[layer_name][sorted_weight_index][num_nodes_to_drop:]).values
        pruned_input_weights = torch.index_select(pruned_model_params[layer_name], -1, keep_index)
        pruned_model_params[layer_name] = pruned_input_weights

        # Go to previous layer and prune when nodes are the output
        prev_layer_index = params_to_index[layer_name] - 1
        prev_layer_name = index_to_params[prev_layer_index]

        if "layernorm" in prev_layer_name:
            pruned_layer_norm_weights = torch.index_select(pruned_model_params[prev_layer_name], 0, keep_index)
            pruned_model_params[prev_layer_name] = pruned_layer_norm_weights

            # Also do bias term
            bias_layer_name = prev_layer_name.replace("weight", "bias")
            pruned_bias_weights = torch.index_select(pruned_model_params[bias_layer_name], 0, keep_index)
            pruned_model_params[bias_layer_name] = pruned_bias_weights

            prev_layer_index = prev_layer_index - 1
            prev_layer_name = index_to_params[prev_layer_index]

        pruned_output_weights = torch.index_select(pruned_model_params[prev_layer_name], 0, keep_index)
        pruned_model_params[prev_layer_name] = pruned_output_weights

        # Also do bias term
        bias_layer_name = prev_layer_name.replace("weight", "bias")
        pruned_bias_weights = torch.index_select(pruned_model_params[bias_layer_name], 0, keep_index)
        pruned_model_params[bias_layer_name] = pruned_bias_weights

torch.save(pruned_model_params, save_name)
state_dict_shapes = {}
for param_name in pruned_model_params.keys():
    state_dict_shapes[param_name] = pruned_model_params[param_name].shape
    #print(param_name, pruned_model_params[param_name].shape)

pkl.dump(state_dict_shapes, open(state_dict_save_name, "wb"))
print(node_num / len(all_nodes))
print(node_num)
