import torch
import tqdm
import torch.nn.functional as F
import copy
import numpy as np


def layer_cmp_scores(output_blocks_list, train_data):
    _, y = train_data
    classes = torch.unique(y)
    output_errors = []
    for block in output_blocks_list:
        class_errors = []
        for i in classes.tolist():
            idxs = (y == i).nonzero(as_tuple=True)[0]
            block_select = torch.index_select(block, 0, idxs)
            block_select_norm = F.normalize(block_select)
            mean_pt = torch.mean(block_select_norm, dim=0)
            loss = F.mse_loss(block_select_norm, mean_pt)
            class_errors.append(loss)
        output_errors.append(np.mean(class_errors))
    return torch.tensor(output_errors)


def path_norm_scores(model, param_blocks_list):
    new_mdl = copy.deepcopy(model)
    counts = [sum([torch.numel(p) for p in param_block]) for param_block in param_blocks_list]
    upper_quartiles = []
    for param_block in param_blocks_list:
        all = torch.cat([torch.flatten(p) for p in param_block])
        upper_quartiles.append(torch.quantile(all, 0.9)**2)
    with torch.no_grad():
        for param in new_mdl.parameters():
            param.multiply_(param)
        out1sum = torch.sum(new_mdl.layer1(torch.ones((1, 16, 32, 32)))) / counts[0] / upper_quartiles[0]
        out2sum = torch.sum(new_mdl.layer2(torch.ones((1, 16, 32, 32)))) / counts[1] / upper_quartiles[1]
        out3sum = torch.sum(new_mdl.layer3(torch.ones((1, 32, 16, 16)))) / counts[2] / upper_quartiles[2]
        x = model.avgpool(model.relu(model.bn(torch.ones((1, 64, 8, 8)))))
        out4sum = torch.sum(new_mdl.fc(x.view(x.size(0), -1))) / counts[3] / upper_quartiles[3]
    return torch.tensor([out1sum, out2sum, out3sum, out4sum])


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
