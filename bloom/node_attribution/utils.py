def count_params(model):
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += count_layer_params(param)
        
    if hasattr(model, "lm_head"):
        base_model_params = total_params
        lm_head_weights = next(param for param in model.lm_head.parameters())
        total_params += count_layer_params(lm_head_weights)
        
        return total_params, base_model_params
        
    return total_params

def count_layer_params(param):
    shape = param.shape
    layer_total = 1

    for dim in shape:
        layer_total *= dim
        
    return layer_total