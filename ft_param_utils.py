import torch


def grad_var_scores(param_blocks_list):
    grad_vars = []
    for param_block in param_blocks_list:
        param_grads = [torch.flatten(param.grad) for param in param_block]
        grad_vars.append(torch.var(torch.cat(param_grads)))
    return torch.tensor(grad_vars)

def get_all_param_blocks(model):
    params_block1 = []
    params_block2 = []
    params_block3 = []
    params_block4 = []
    for name, param in model.named_parameters():
        if name.startswith("layer1"):
            params_block1.append(param)
        elif name.startswith("layer2"):
            params_block2.append(param)
        elif name.startswith("layer3"):
            params_block3.append(param)
        elif name.startswith("bn") or name.startswith("fc"):
            params_block4.append(param)
    return [params_block1, params_block2, params_block3, params_block4]
