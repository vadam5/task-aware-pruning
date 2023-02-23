from node_attribution.node_attribution import get_attributions

model_size = "560m"
data = ["Hello, I am an AlexPrize chatbot"]
prune_percent = 0.30

# Get attributions
avg_contributions, max_contributions, model = get_attributions(model_size, data)

# Greedy prune
