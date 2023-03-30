import random
import torch
import pickle as pkl

from transformers import AutoTokenizer, BloomForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

from transformers.models.bloom.configuration_bloom import BloomConfig
from pruning.pruned_bloom import PrunedBloomForCausalLM


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3B")
context_length = 2048

print("Loading pruned model raw")
pruned_model = BloomForCausalLM.from_pretrained("bigscience/bloom-3B")
print("Moving model to CUDA")
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
torch.save(finetuned_model, "finetuned_full_bloom3B.pt")
