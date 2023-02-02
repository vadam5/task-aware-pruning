from transformers import AutoTokenizer, BloomForCausalLM
from bloom_for_node_attribution import BloomForCausalLMForNodeAttribution

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLMForNodeAttribution.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Hello, I'm am conscious and", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))