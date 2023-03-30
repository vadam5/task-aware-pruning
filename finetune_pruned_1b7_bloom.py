import random
import torch
import pickle as pkl

from transformers import AutoTokenizer, BloomForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

from transformers.models.bloom.configuration_bloom import BloomConfig
from pruning.pruned_bloom import PrunedBloomForCausalLM

model_size = "1b7"
tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloom-{model_size}")
context_length = 2048

# Load model
weights_path = f"../models/pruned_40percent_{model_size}_bloom.pt"
state_dict_shapes_path = f"../models/pruned_40percent_{model_size}_bloom_state_dict_shapes.pkl"

bloom_config = BloomConfig(
    vocab_size=250880,
    hidden_size=2048,
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
print("Loading pruned model raw")
pruned_model = PrunedBloomForCausalLM(bloom_config, state_dict_shapes_path)
print("Loading pruned model weights")
pruned_model.load_state_dict(torch.load(weights_path))
print("moving pruned model to device")
pruned_model = pruned_model.cuda()

#Load data
print("Loading data")
train_data = pkl.load(open("../data/cleaned_3353_pairs.pkl", "rb"))
val_data = pkl.load(open("../data/530_human_filtered_conv_pairs.pkl", "rb"))
val_data = [line for line in val_data if line not in set(train_data)]

print(len(val_data))
random.shuffle(train_data)

# tokenize data
tokenizer.pad_token = tokenizer.eos_token

def tokenize(data, tokenizer, context_length):
    outputs = tokenizer(
        data,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length < context_length:
            input_batch.append(input_ids)

    return input_batch


class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, context_length):
        self.data_strings = data_list
        self.tokenized_data = tokenize(data_list, tokenizer, context_length)

    def __len__(self):
        return len(self.data_strings)

    def __getitem__(self, idx):
        return self.tokenized_data[idx].copy()

print("tokenizing data set")
train_dataset = DialogueDataset(train_data, tokenizer, context_length)
val_dataset = DialogueDataset(val_data, tokenizer, context_length)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Setup trainer args
print("init trainer")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=6,
    logging_steps=5,
)

# init trainer
trainer = Trainer(
    model=pruned_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
print("starting training")
trainer.train()

print("Saving model")
trainer.save_model()
finetuned_model = trainer.model

print("Saving model again")
torch.save(finetuned_model, "finetuned_50percent_pruned_bloom3B.pt")
