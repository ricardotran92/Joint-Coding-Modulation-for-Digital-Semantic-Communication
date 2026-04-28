import torchvision
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from network import JCM
from analog_network import AnalogNet
from train import train
from train_analog import train_analog
from evaluation import EVAL
from utils import init_seeds
from network import canonical_mod_method
import os
import argparse
import matplotlib.pyplot as plt


def inspect_one_sample(net, data_loader, device):
    net.eval()
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        code, z, z_hat, r_class, recon, debug = net(data, return_debug=True)

    idx = 0
    pred = torch.argmax(r_class[idx]).item()
    print("\n=== Inspect One Sample ===")
    print(f"target label: {target[idx].item()}, predicted label: {pred}")
    print(f"X shape: {tuple(data[idx].shape)}")
    print(f"h shape: {tuple(debug['h'][idx].shape)}")
    print(f"q shape: {tuple(debug['q'][idx].shape)}")
    print(f"Z shape: {tuple(debug['z_discrete'][idx].shape)}")
    print(f"Z^ shape: {tuple(debug['z_hat'][idx].shape)}")
    print(f"X^ shape: {tuple(debug['x_hat'][idx].shape)}")
    print(f"S^ shape: {tuple(debug['s_hat'][idx].shape)}")

    print("\nX[0, :2, :2]:")
    print(data[idx, 0, :2, :2].detach().cpu())
    print("\nh[:12]:")
    print(debug['h'][idx, :12].detach().cpu())
    print("\nq[0:5]:")
    print(debug['q'][idx, :5].detach().cpu())
    print("\nZ[0:12]:")
    print(debug['z_discrete'][idx, :12].detach().cpu())
    print("\nZ^[0:12]:")
    print(debug['z_hat'][idx, :12].detach().cpu())
    print("\nS^[:10]:")
    print(debug['s_hat'][idx, :10].detach().cpu())

    z_cpu = debug["z_discrete"][idx].detach().cpu().flatten()
    z_hat_cpu = debug["z_hat"][idx].detach().cpu().flatten()
    n = (z_cpu.numel() // 2) * 2
    if n >= 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(z_cpu[:n:2], z_cpu[1:n:2], s=10, alpha=0.8, label="Z (clean)")
        plt.scatter(z_hat_cpu[:n:2], z_hat_cpu[1:n:2], s=10, alpha=0.6, label="Z^ (noisy)")
        plt.title("Constellation of One Sample")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.show()


def canonical_mode(mode):
    value = str(mode).strip().lower()
    while len(value) >= 2 and (
        (value[0] == "'" and value[-1] == "'") or
        (value[0] == '"' and value[-1] == '"')
    ):
        value = value[1:-1].strip().lower()
    return value


def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


def main(config):
    # initialize random seed
    init_seeds()
    config.mod_method = canonical_mod_method(config.mod_method)
    config.mode = canonical_mode(config.mode)

    # prepare training & test data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform_train,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = JCM(config, device).to(device)

    if config.pretrain_analog:
        pretrain_net = AnalogNet(config, device).to(device)

        print("Pretraining the analog network.")
        train_analog(config, pretrain_net, train_loader, test_loader, device)

    else:
        if config.mode == 'train':
            print("Loading the pretrained analog network...")

            if config.mod_method == 'bpsk':
                model_name = '/pretrained_analog/1d/CIFAR_analog_SNR{:.3f}_Trans{:d}.pth.tar'.format(
                             config.snr_train, config.channel_use)
            else:
                model_name = '/pretrained_analog/2d/CIFAR_analog_SNR{:.3f}_Trans{:d}.pth.tar'.format(
                             config.snr_train, config.channel_use)
                # the output dimension of BPSK is 1/2 of that of M-QAM

            pretrained_dict = torch.load('./checkpoints' + model_name, map_location=torch.device('cpu'))
            model_dict = net.state_dict()
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict, strict=False)
            print('Successfully load the pretrained analog model!')

            print("Training with the modulation scheme {}.".format(config.mod_method))
            train(config, net, train_loader, test_loader, device)

        elif config.mode == 'test':
            print("Start Testing.")

            if config.load_checkpoint:
                model_name = '/{}/'.format(config.mod_method) + \
                             'CIFAR_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                                 config.snr_train, config.channel_use, config.mod_method)
                net.load_state_dict(torch.load('./checkpoints' + model_name, map_location=torch.device('cpu')))

            if getattr(config, "inspect_steps", 0):
                inspect_one_sample(net, test_loader, device)

            acc, mse, psnr, ssim = EVAL(net, test_loader, device, config)
            print('acc: {:.3f}, mse: {:3f}, psnr: {:.3f}, ssmi: {:.3f}'.format(acc, mse, psnr, ssim))

        else:
            print("Wrong mode input!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=128)
    """Available modulation methods:"""
    """bpsk, 4qam, 16qam, 64qam"""
    parser.add_argument('--mod_method', type=str, default='4qam')
    parser.add_argument('--load_checkpoint', type=int, default=1)
    parser.add_argument('--pretrain_analog', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snr_train', type=float, default=12)
    parser.add_argument('--snr_test', type=float, default=12)
    """The tradeoff hyperparameter lambda between two tasks"""
    parser.add_argument('--tradeoff_lambda', type=float, default=200)

    # misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--inspect_steps', type=int, default=0)

    config = parser.parse_args()

    mischandler(config)
    main(config)
