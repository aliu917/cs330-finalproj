import torch.nn.functional as F
import torch
import tqdm
import wandb

import learn_ft_params
import utils

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'epochs': {'values': [15]},
        'lr': {'values': [1e-3, 1e-4, 1e-5]},
        'noise_block': {'values': [1, 2, 3, 4]},
        'random_degree': {'values': [0.1]},
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='cs330-finalproj')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def learn_and_freeze_model_params(model, target_block_number):
    pred_block_number = learn_ft_params.predict_block(model, target_block_number)
    for name, param in model.named_parameters():
        if (pred_block_number > 0 and pred_block_number <= 3) and name.startswith("layer" + str(pred_block_number)):
            # https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
            continue
        elif (pred_block_number == 4) and (name.startswith("bn") or name.startswith("fc")):
            param.grad = torch.zeros_like(param)
        else:
            param.grad = None
        x = 5


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
    add_model_noise_to_block(model, wandb.config.noise_block, wandb.config.random_degree)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    train_data, val_data = utils.get_data()

    wandb.watch(model, log="all", log_freq=1)
    for epoch in range(wandb.config.epochs):
        train_correct = []
        train_losses = []

        model.train()
        for i, (x,y) in tqdm.tqdm(enumerate(train_data)):
            optimizer_custom_zero_grad(model)
            pred_y = model(x)
            loss = F.cross_entropy(pred_y, y)
            train_losses.append(loss.cpu())
            _, predicted = pred_y.max(1)
            train_correct.append(predicted.eq(y).cpu())
            loss.backward()
            learn_and_freeze_model_params(model, wandb.config.noise_block)
            optimizer.step()

        num_correct = torch.count_nonzero(torch.cat(train_correct)).item()
        train_loss = torch.mean(torch.tensor(train_losses, device=device)).item()
        train_acc = num_correct / len(torch.cat(train_correct))
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
                   "val_acc": val_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id, function=train, count=3)