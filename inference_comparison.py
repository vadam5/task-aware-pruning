import torch
import json
import time
import statistics
import numpy as np
import pickle as pkl

from tqdm import tqdm


from transformers import AutoTokenizer, BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
from pruning.pruned_bloom import PrunedBloomForCausalLM
from node_attribution.utils import count_params

tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-560m")

data = pkl.load(open("../data/44_human_filtered_conv_pairs.pkl", "rb"))
cali_data = data[:22]
val_data = data[22:]

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    tensor_input = tensor_input.cuda()
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    mask = mask.cuda()
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.pad_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.pad_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


def calc_p(data):
    perplexity_sum = 0
    for pair in tqdm(data):
        perplexity = score(sentence=pair, model=pruned_model, tokenizer=tokenizer)
        perplexity_sum += perplexity

    p = perplexity_sum / len(val_data)

    return p

path = "../models/pruned_10percent_1b7_bloom_full.pt"
pruned_model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
#pruned_model = torch.load(path)
pruned_model = pruned_model.cuda()
#pruned_model = PrunedBloomForCausalLM(bloom_config, state_dict_shapes_path)
#pruned_model.load_state_dict(torch.load(weights_path))

pruned_percent = 1.0 - (count_params(pruned_model)[-1] / 559214592)
print(pruned_percent)
num_trials = 20
line = "Hello, I am a social bot! "
inputs = tokenizer(line, return_tensors="pt")
inputs = inputs["input_ids"].cuda()
pruned_times = []

for i in range(num_trials):
    start = time.time()
    outputs = pruned_model.generate(
        input_ids=inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
    )
    end = time.time()
    pruned_times.append(end - start)
    print(f"inference time: {end - start}")
    
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

