import torch.nn.functional as F
import torch
import tqdm
import wandb
import numpy as np
import copy
import mann
import random

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
        'pred_type': {'values': ["inner_loop_full_step_mlp"]},
        'inner_batch': {'values': [4]}
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
    return model


def predict_block(model, target_block_number, pred_type, train_data, multipliers, inner_model):
    if pred_type == "correct":
        return target_block_number
    param_blocks_list = ft_param_utils.get_all_param_blocks(model)
    output_blocks_list = [activation['layer' + str(i)] for i in range(1, 5)]
    block_scores = torch.zeros(4)
    if "grad_mag" in pred_type:
        block_scores = ft_param_utils.grad_mag_scores(param_blocks_list)
    elif "grad_var" in pred_type:
        block_scores = ft_param_utils.grad_var_scores(param_blocks_list)
    elif "path_norm" in pred_type:
        block_scores = ft_param_utils.path_norm_scores(model, param_blocks_list)
    elif "layer_cmp" in pred_type:
        block_scores = ft_param_utils.layer_cmp_scores(output_blocks_list, train_data)
    elif "mann" in pred_type:
        block_scores = predict_inner_block(inner_model, output_blocks_list, train_data)
    elif "mlp" in pred_type:
        preds = torch.argmax(model.inner_forward(train_data[0]), dim=1)
        block_scores = torch.zeros(4)
        for i in range(4):
            block_scores[i] = torch.count_nonzero(preds == i).item() / len(preds)
    print("block", target_block_number, "scores: ", block_scores)
    if "layer_cmp" in pred_type:
        block_scores = block_scores * multipliers
        print("block", target_block_number, "FIXED scores: ", block_scores)
    return torch.argmax(block_scores).item() + 1


def calibrate_model(model, train_data):
    model.eval()
    all_scores = []
    for i, (x, y) in tqdm.tqdm(enumerate(train_data)):
        model(x)
        param_blocks_list = ft_param_utils.get_all_param_blocks(model)
        output_blocks_list = [activation['layer' + str(i)] for i in range(1, 5)]
        if "grad_var" in wandb.config.pred_type:
            block_scores = ft_param_utils.grad_var_scores(param_blocks_list)
        elif "path_norm" in wandb.config.pred_type:
            block_scores = ft_param_utils.path_norm_scores(model, param_blocks_list)
        elif "layer_cmp":
            block_scores = ft_param_utils.layer_cmp_scores(output_blocks_list, (x, y))
        all_scores.append(block_scores)
    avgs = torch.mean(torch.cat([block.unsqueeze(1) for block in all_scores], dim=1), dim=1)
    print("block avgs: ", avgs)
    return 0.25 / avgs


def learn_and_freeze_model_params(model, target_block_number, train_data, multipliers, inner_model):
    pred_block_number = predict_block(model, target_block_number, wandb.config.pred_type, train_data, multipliers, inner_model)
    # print("predicted block: ", pred_block_number, " actual: ", target_block_number)
    freeze_model_params(model, pred_block_number)
    return pred_block_number


def freeze_model_params(model, unfrozen_block):
    for name, param in model.named_parameters():
        if (unfrozen_block > 0 and unfrozen_block <= 3) and name.startswith("layer" + str(unfrozen_block)):
            # https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
            continue
        elif (unfrozen_block == 4) and (name.startswith("bn") or name.startswith("fc")):
            continue
        elif name.startswith("inner_mlp"):
            continue
        else:
            param.grad = None


def optimizer_custom_zero_grad(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param)


def train():
    run = wandb.init()
    print(wandb.config)

    model = utils.get_model()
    orig_model = copy.deepcopy(model)
    inner_model = None
    if "mann" in wandb.config.pred_type:
        input_dim = 3 * 32 * 32 + 16 * 32 * 32 + 32 * 16 * 16 + 64 * 8 * 8 + 10
        inner_model = utils.get_inner_model(input_dim)
    register_model_hooks(model)
    # if not "inner_loop" in wandb.config.pred_type:
    #     utils.add_model_noise_to_block(model, wandb.config.noise_block, wandb.config.random_degree)
    utils.add_model_noise_to_block(model, wandb.config.noise_block, wandb.config.random_degree)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    if "mann" in wandb.config.pred_type:
        inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=wandb.config.lr)

    train_data, val_data, few_shot_train = utils.get_data(pred_type=wandb.config.pred_type)

    # Calibrate so all equal
    multipliers = calibrate_model(model, train_data)
    if "mlp" in wandb.config.pred_type:
        pretrain_mlp(orig_model, train_data)

    wandb.watch(model, log="all", log_freq=1)
    for epoch in range(wandb.config.epochs):
        train_correct = []
        train_losses = []
        block_preds = []
        inner_accs = []

        model.train()
        for i, (x,y) in tqdm.tqdm(enumerate(train_data)):
            if "mann" in wandb.config.pred_type:
                x_inner = x[:wandb.config.inner_batch, :, :, :]
                x = x[wandb.config.inner_batch:, :, :, :]
                y = y[wandb.config.inner_batch:]
                model_copies = []
                for i in range(4):
                    model_copy = copy.deepcopy(orig_model)
                    model_copy_noisy = utils.add_model_noise_to_block(model_copy, i + 1, wandb.config.random_degree)
                    model_copies.append(register_model_hooks(model_copy_noisy))
                inner_acc = inner_train_mann(inner_model, model_copies, x_inner, inner_optimizer)
                inner_accs.append(inner_acc)

            optimizer_custom_zero_grad(model)
            pred_y = model(x)
            loss = F.cross_entropy(pred_y, y)
            train_losses.append(loss.cpu())
            _, predicted = pred_y.max(1)
            train_correct.append(predicted.eq(y).cpu())
            loss.backward()
            if wandb.config.pred_type != "none":
                pred_block = learn_and_freeze_model_params(model, wandb.config.noise_block, (x,y), multipliers, inner_model)
            else:
                pred_block = 0
            block_preds.append(pred_block)
            optimizer.step()
            if "mann" in wandb.config.pred_type:
                inner_optimizer.step()

        if "mann" in wandb.config.pred_type:
            inner_total_accs = np.sum(inner_accs) / len(inner_accs)
            print("inner accs: ", inner_total_accs)
            wandb.log({"inner accs: ": inner_total_accs})
        print("predicted blocks: ", block_preds)
        block_acc = np.count_nonzero(np.array(block_preds) == wandb.config.noise_block) / len(block_preds)
        block1 = np.count_nonzero(np.array(block_preds) == 1) / len(block_preds)
        block2 = np.count_nonzero(np.array(block_preds) == 2) / len(block_preds)
        block3 = np.count_nonzero(np.array(block_preds) == 3) / len(block_preds)
        block4 = np.count_nonzero(np.array(block_preds) == 4) / len(block_preds)
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
                   "block_acc": block_acc,
                   "block1": block1,
                   "block2": block2,
                   "block3": block3,
                   "block4": block4})


def inner_train_mann(inner_model, outer_model_copies, x, inner_optimizer, num_shots=3, train_steps=10):
    import time

    times = []
    accs = []
    for step in range(train_steps):
        ## Sample Batch
        t0 = time.time()
        i, l = sample_inner_data(outer_model_copies, x, num_shots)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        pred, ls = mann.train_step(i, l, inner_model, inner_optimizer)
        t2 = time.time()
        print("[Inner] Loss/train", ls, step)
        wandb.log({"inner train loss": ls})
        pred = torch.reshape(
            pred, [-1, 4, 4, 4]
        )
        pred = torch.argmax(pred[:, -1, :, :], axis=2)
        l = torch.argmax(l[:, -1, :, :], axis=2)
        acc = pred.eq(l).sum().item() / 4
        accs.append(acc)
        times.append([t1 - t0, t2 - t1])

    avg_accs = np.sum(accs) / len(accs)
    return avg_accs

        # ## Evaluate
        # if (step + 1) % config.eval_freq == 0:
        #     print("[Inner] " + "*" * 5 + "Iter " + str(step + 1) + "*" * 5)
        #     i, l = next(test_loader)
        #     i, l = i.to(device), l.to(device)
        #     pred, tls = train_step(i, l, model, optim, eval=True)
        #     print("[Inner] Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
        #     wandb.log({"inner train loss": ls.cpu().numpy(),
        #                "inner test loss": tls.cpu().numpy()})
        #     pred = torch.reshape(
        #         pred, [-1, config.num_shot + 1, config.num_classes, config.num_classes]
        #     )
        #     pred = torch.argmax(pred[:, -1, :, :], axis=2)
        #     l = torch.argmax(l[:, -1, :, :], axis=2)
        #     acc = pred.eq(l).sum().item() / (config.meta_batch_size * config.num_classes)
        #     print("Test Accuracy", acc)
        #     writer.add_scalar("Accuracy/test", acc, step)
        #
        #     times = np.array(times)
        #     print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
        #     times = []
        #     # scheduler.step(tls)
        #     print(f"learning rate {optim.param_groups[0]['lr']}")


def pretrain_mlp(model, train_data):
    model_copies = []
    for i in range(4):
        model_copy = copy.deepcopy(model)
        model_copy_noisy = utils.add_model_noise_to_block(model_copy, i + 1, wandb.config.random_degree)
        model_copies.append(register_model_hooks(model_copy_noisy))

    model.train()
    optimizer = torch.optim.Adam(model.inner_mlp.parameters(), lr=wandb.config.lr)
    freeze_model_params(model, 0)  # 0 is not a valid block so all original blocks will be frozen
    for epoch in range(15):
        blocks_acc = [[0], [0], [0], [0]]
        for i, (x, y) in tqdm.tqdm(enumerate(train_data)):
            model_cpy_idx = random.choice(range(4))
            model_copies[model_cpy_idx].inner_forward(x)
            out = model.bn(activation["layer3"])
            out = model.relu(out)
            out = model.avgpool(out)
            out = out.view(out.size(0), -1)
            inner_pred = model.inner_mlp(out)
            loss = F.cross_entropy(inner_pred, torch.ones(inner_pred.shape[0], dtype=torch.int64)*model_cpy_idx)
            loss.backward()
            optimizer.step()
            num_correct = torch.count_nonzero(torch.argmax(inner_pred, axis=1) == model_cpy_idx).item()
            acc = num_correct / inner_pred.shape[0]
            blocks_acc[model_cpy_idx].append(acc)
        for i in range(4):
            block_i_acc = np.sum(blocks_acc[i]) / (len(blocks_acc[i]) - 1)
            print("Inner train acc block " + str(i+1), block_i_acc)
            wandb.log({"Inner acc block " + str(i+1): block_i_acc})


def predict_inner_block(inner_model, output_blocks_list, train_data):
    outputs = torch.cat([torch.flatten(output_blocks_list[i][-1,...]) for i in range(4)])
    outputs_data_combined = torch.cat((torch.flatten(train_data[0][-1,...]), outputs))
    preds = inner_model(outputs_data_combined.unsqueeze(0).unsqueeze(0).unsqueeze(0), torch.zeros(4).unsqueeze(0).unsqueeze(0).unsqueeze(0))
    return preds.squeeze()

def sample_inner_data(outer_model_copies, x, num_shots):
    random_idxs = random.sample(range(x.shape[0]), num_shots + 1)
    x_select = x[random_idxs]

    data = []
    labels = []
    for layer in range(4):
        outer_model = outer_model_copies[layer]
        outer_model(x_select)
        classes_data = [torch.cat([torch.flatten(activation['layer' + str(i)][j]) for i in range(1,5)]) for j in range(num_shots + 1)]
        classes_data_combined = torch.cat((torch.flatten(x_select, start_dim=1), torch.stack(classes_data)), dim=1)
        data.append(classes_data_combined)
        label = torch.zeros(4)
        label[layer] = 1
        labels.append(torch.stack([label for _ in range(num_shots + 1)]))

    data = torch.tensor(torch.stack(data), device=device).permute(1, 0, 2).unsqueeze(0)
    labels = torch.tensor(torch.stack(labels), device=device).permute(1, 0, 2).unsqueeze(0)

    # https://discuss.pytorch.org/t/shuffle-a-tensor-a-long-a-certain-dimension/129798/2
    for i in range(num_shots):
        idx = torch.randperm(data.size(1))
        data[:, i, ...] = data[:, i, idx]
        labels[:, i, ...] = labels[:, i, idx]

    return data, labels

if __name__ == '__main__':
    wandb.agent(sweep_id, function=train, count=4)