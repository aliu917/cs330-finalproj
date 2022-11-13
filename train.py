import torch.nn.functional as F
import torch
import tqdm
import wandb
import numpy as np

import ft_param_utils
import utils

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'epochs': {'values': [15]},
        'lr': {'values': [1e-3]}, #, 1e-4, 1e-5]},
        'noise_block': {'values': [1, 2, 3, 4]},
        'random_degree': {'values': [0.1]},
        'pred_type': {'values': ["grad_var"]},
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='cs330-finalproj')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def register_model_hooks(model):
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.fc.register_forward_hook(get_activation('layer4'))


def predict_block(model, target_block_number, pred_type):
    if pred_type == "correct":
        return target_block_number
    param_blocks_list = ft_param_utils.get_all_param_blocks(model)
    output_blocks_list = [activation['layer' + str(i)] for i in range(1, 5)]
    block_scores = torch.zeros(4)
    if "grad_var" in pred_type:
        block_scores = ft_param_utils.grad_var_scores(param_blocks_list)
    return torch.argmax(block_scores).item() + 1


def learn_and_freeze_model_params(model, target_block_number):
    pred_block_number = predict_block(model, target_block_number, wandb.config.pred_type)
    # print("predicted block: ", pred_block_number, " actual: ", target_block_number)
    for name, param in model.named_parameters():
        if (pred_block_number > 0 and pred_block_number <= 3) and name.startswith("layer" + str(pred_block_number)):
            # https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
            continue
        elif (pred_block_number == 4) and (name.startswith("bn") or name.startswith("fc")):
            continue
        else:
            param.grad = None
    return pred_block_number


def optimizer_custom_zero_grad(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param)


def add_model_noise_to_block(model, block_number, random_degree):
    with torch.no_grad():
        if block_number > 0 and block_number <= 3:
            for name, param in model.named_parameters():
                if name.startswith("layer" + str(block_number)):
                    param.add_(torch.randn(param.shape) * random_degree)
        elif block_number == 4:
            for name, param in model.named_parameters():
                if name.startswith("bn") or name.startswith("fc"):
                    param.add_(torch.randn(param.shape) * random_degree)


def train():
    run = wandb.init()
    print(wandb.config)

    model = utils.get_model()
    register_model_hooks(model)
    add_model_noise_to_block(model, wandb.config.noise_block, wandb.config.random_degree)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    train_data, val_data = utils.get_data()

    wandb.watch(model, log="all", log_freq=1)
    for epoch in range(wandb.config.epochs):
        train_correct = []
        train_losses = []
        block_preds = []

        model.train()
        for i, (x,y) in tqdm.tqdm(enumerate(train_data)):
            optimizer_custom_zero_grad(model)
            pred_y = model(x)
            loss = F.cross_entropy(pred_y, y)
            train_losses.append(loss.cpu())
            _, predicted = pred_y.max(1)
            train_correct.append(predicted.eq(y).cpu())
            loss.backward()
            pred_block = learn_and_freeze_model_params(model, wandb.config.noise_block)
            block_preds.append(pred_block)
            optimizer.step()

        block_acc = np.count_nonzero(np.array(block_preds) == wandb.config.noise_block) / len(block_preds)
        num_correct = torch.count_nonzero(torch.cat(train_correct)).item()
        train_loss = torch.mean(torch.tensor(train_losses, device=device)).item()
        train_acc = num_correct / len(torch.cat(train_correct))
        print("Block acc: ", block_acc)
        print("Finished Epoch", epoch, ", training loss:", train_loss, ", training acc: ", train_acc)

        with torch.no_grad():
            model.eval()
            val_correct = []
            val_losses = []

            for i, (x,y) in tqdm.tqdm(enumerate(val_data)):
                pred_y = model(x)
                loss = F.cross_entropy(pred_y, y)
                val_losses.append(loss.cpu())
                _, predicted = pred_y.max(1)
                val_correct.append(predicted.eq(y).cpu())

            num_correct = torch.count_nonzero(torch.cat(val_correct)).item()
            val_loss = torch.mean(torch.tensor(val_losses, device=device)).item()
            val_acc = num_correct / len(torch.cat(val_correct))
            print("Finished Epoch", epoch, ", val loss:", val_loss, ", val acc: ", val_acc)

        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "train_acc": train_acc,
                   "val_loss": val_loss,
                   "val_acc": val_acc,
                   "block_acc": block_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id, function=train, count=4)