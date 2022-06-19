from torch.utils.data import DataLoader
import torch
from modules.modules import *
from datasets.dataset import IKDataset, IKDatasetVal
import ikpy.chain
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json


def train(cfg):
    r_arm = ikpy.chain.Chain.from_urdf_file(cfg.chain_path)

    upper = []
    lower = []
    for i in range(1, len(r_arm.links) - 1):
        lower.append(r_arm.links[i].bounds[0])
        upper.append(r_arm.links[i].bounds[1])

    upper = np.array(upper)
    lower = np.array(lower)

    train_dataset = IKDataset(cfg.train_data_path)
    test_dataset = IKDatasetVal(cfg.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    hypernet = HyperNet(cfg).cuda()

    mainnet = MainNet(cfg).cuda()

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=cfg.lr)

    train_counter, test_counter = 0, 0
    train_loss, test_loss = 0, 0

    best_test_loss = np.inf
    best_test_epoch = 0

    epochs_without_improvements = 0

    train_losses = []
    test_losses = []

    for epoch in range(cfg.num_epochs):
        hypernet.train()
        for positions, joint_angles in train_dataloader:
            positions, joint_angles = positions.cuda(), joint_angles.cuda()
            output = torch.cat((torch.ones(joint_angles.shape[0], 1).cuda(), joint_angles), dim=1)

            optimizer.zero_grad()
            predicted_weights = hypernet(positions)
            distributions, selection = mainnet(output, predicted_weights)

            losses = [-torch.mean(distributions[i].log_prob(joint_angles[:, i].unsqueeze(1))) for i in range(len(distributions))]

            loss = sum(losses) / len(losses)

            train_counter += 1
            train_loss += loss.item()

            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), cfg.grad_clip)

            optimizer.step()

        train_losses.append(train_loss / train_counter)
        print(f"Train loss {train_losses[-1]}")

        train_loss, train_counter = 0, 0

        sampled = []

        hypernet.eval()
        for test_iter, (positions, joint_angles) in enumerate(test_dataloader):
            positions, joint_angles = positions.cuda(), joint_angles.cuda()
            predicted_weights = hypernet(positions)

            for j in range(cfg.num_solutions_validation):
                samples, distributions, means,variance, selection = mainnet.validate(torch.ones(joint_angles.shape[0], 1).cuda(), predicted_weights, lower, upper)
                sampled.append(samples)

        for sampled_lst in sampled:
            for k in range(len(positions)):
                joint_angles = [0] + [sampled_lst[i][k].item() for i in range(cfg.num_joints)] + [0]
                real_frame = r_arm.forward_kinematics(joint_angles)
                test_loss += np.sqrt(np.sum((real_frame[:3, 3] - positions[k].detach().cpu().numpy()) ** 2))
                test_counter += 1

        test_losses.append(test_loss / test_counter)
        print(f"Test loss {test_losses[-1]}")

        if test_losses[-1] < best_test_loss:
            epochs_without_improvements = 0
            best_test_loss = test_losses[-1]
            torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/best_model.pt')
            torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/best_optimizer.pt')
            with open(f'{cfg.exp_dir}/best_test_loss.txt', 'a+') as f:
                f.write(f'Epoch {epoch} - test loss {best_test_loss} \n')
        else: #advance early stopping counter
            epochs_without_improvements += 1

        if epochs_without_improvements == cfg.early_stopping_epochs:
            break

        test_loss, test_counter = 0, 0

        plt.plot(range(len(train_losses)), train_losses, label = 'train')
        plt.savefig(f'{cfg.exp_dir}/train_plot.png')
        plt.clf()
        plt.plot(range(len(test_losses)), test_losses, label = 'test')
        plt.savefig(f'{cfg.exp_dir}/test_plot.png')
        plt.clf()


        torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/last_model.pt')
        torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/last_optimizer.pt')


def create_exp_dir(cfg):
    existing_dirs = os.listdir(cfg.exp_dir)
    if existing_dirs:
        sorted_dirs = sorted(existing_dirs, key=lambda x : int(x.split('_')[1]))
        last_exp_num = int(sorted_dirs[-1].split('_')[1])
        exp_name = f"{cfg.exp_dir}/exp_{last_exp_num + 1}"
    else:
        exp_name = f"{cfg.exp_dir}/exp_0"
    os.makedirs(exp_name)
    with open(f'{exp_name}/run_args.json', 'w+') as f:
        json.dump(cfg.__dict__, f, indent=2)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain-path', type=str, default="/home/raphael/home/raphael/PycharmProjects/hyperikArxiv/assets/digit/urdf/digit_r_arm.urdf", help='urdf chain path')
    parser.add_argument('--train-data-path', type=str, default="/home/raphael/home/raphael/PycharmProjects/hyperikArxiv/data/digit/train_20000.hdf5", help='urdf chain path')
    parser.add_argument('--test-data-path', type=str, default='/home/raphael/home/raphael/PycharmProjects/hyperikArxiv/data/digit/val_1000.hdf5', help='urdf chain path')
    parser.add_argument('--num-joints', type=int, default=4, help='number of joints of the kinematic chain')


    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=100000, help='learning rate')
    parser.add_argument('--num-solutions-validation', type=int, default=10, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--early-stopping-epochs', type=int, default=50, help='number of epochs without improvement to trigger end of training')
    parser.add_argument('--grad-clip', type=int, default=1, help='clip norm of gradient')
    parser.add_argument('--embedding-dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--hypernet-input-dim', type=int, default=3, help='number of input to the hypernetwork (f). default case 3 (x, y, z)')
    parser.add_argument('--hypernet-hidden-size', type=int, default=1024, help='hypernetwork (f) number of neurons in hidden layer')
    parser.add_argument('--hypernet-num-hidden-layers', type=int, default=3, help='hypernetwork  (f) number of hidden layers')
    parser.add_argument('--jointnet-hidden-size', type=int, default=256, help='jointnet (g) number of neurons in hidden layer')
    parser.add_argument('--num-gaussians', type=int, default=50, help='number of gaussians for mixture . default=1 no mixture')
    parser.add_argument('--exp_dir', type=str, default='runs', help='folder path name to save the experiment')

    parser.set_defaults()
    cfg = parser.parse_args()

    full_exp_dir = create_exp_dir(cfg)

    cfg.exp_dir = full_exp_dir

    cfg.jointnet_output_dim = cfg.num_gaussians * 2 + cfg.num_gaussians if cfg.num_gaussians != 1 else 2

    train(cfg)



