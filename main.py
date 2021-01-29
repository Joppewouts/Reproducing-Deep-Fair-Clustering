# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import torch
from torch import nn

from dataloader import get_dataset
from kmeans import get_cluster_centers
from module import Encoder
from adverserial import adv_loss
from eval import predict, cluster_accuracy, balance
from utils import set_seed, AverageMeter, target_distribution, aff, inv_lr_scheduler
import os
import wandb
from vae import DFC_VAE
from vae import train as train_vae
from dfc import train as train_dfc
from dec import train as train_dec
from dfc import DFC
from resnet50_finetune import *
import torchvision.models as models

import pytorch_lightning as pl
from pl_bolts.models.autoencoders import VAE


def get_encoder(args, log_name, legacy_path, path, dataloader_list, device='cpu', encoder_type='vae'):
    if encoder_type == 'vae':
        print('Loading the variational autoencoder')
        if legacy_path:
            encoder = Encoder().to(device)
            encoder.load_state_dict(torch.load(
                legacy_path, map_location=device))
        else:
            if path:
                model = DFC_VAE.load_from_checkpoint(path).to(device)
            else:
                model = train_vae(args, log_name,  dataloader_list, args.input_height,
                                  is_digit_dataset=args.digital_dataset, device=device).to(device)
            encoder = model.encoder
    elif encoder_type == 'resnet50':  # Maybe fine tune resnet50 here
        print('Loading the RESNET50 encoder')
        encoder = models.resnet50(pretrained=True, progress=True)
        
        set_parameter_requires_grad(encoder, req_grad=False)
        # encoder.fc = nn.Linear(1000, args.dfc_hidden_dim) #TODO: Reshape and finetune resnet50
        # get_update_param(encoder)
        encoder = encoder.to(device)
        # encoder, val_acc_history = train_last_layer_resnet50( #train for the 31 classes
            # encoder, dataloader_list, log_name=log_name, device=device, args=args, num_classes=args.dfc_hidden_dim)

    else:
        raise NameError('The encoder_type variable has an unvalid value')
    wandb.watch(encoder)
    return encoder


def get_dec(args, path, dataloader_list, encoder, save_name, device='cpu', centers=None):
    if path:
        dec = DFC(cluster_number=args.cluster_number,
                  hidden_dimension=args.dfc_hidden_dim).to(device)
        dec.load_state_dict(torch.load(path, map_location=device))
    else:
        dec = train_dec(args, dataloader_list, encoder, device,
                        centers=centers,  save_name=save_name)
    return dec


def get_dfc(args, path, dataloader_list, encoder, save_name, encoder_group_0=None, encoder_group_1=None, dfc_group_0=None, dfc_group_1=None, device='cpu', centers=None, get_loss_trade_off=lambda step: (10, 10, 10)):
    if path:
        dfc = DFC(cluster_number=args.cluster_number,
                  hidden_dimension=args.dfc_hidden_dim).to(device)
        dfc.load_state_dict(torch.load(path, map_location=device))
    else:
        dfc = train_dfc(args, dataloader_list, encoder, encoder_group_0, encoder_group_1, dfc_group_0, dfc_group_1,
                        device, centers=centers, get_loss_trade_off=get_loss_trade_off, save_name=save_name)
    return dfc


def main(args):
    set_seed(args.seed)

    # Use float16 tensor for memory efficiency. (There is also an if statement in the dataloader)
    if args.half_tensor:
        torch.set_default_tensor_type('torch.HalfTensor')
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print(f"Using {device}")

    dataloader_0, dataloader_1 = get_dataset[args.dataset](args)

    print("Loading Encoder")
    encoder = get_encoder(args, "encoder", args.encoder_legacy_path, args.encoder_path, [
                          dataloader_0, dataloader_1], device=device, encoder_type=args.encoder_type)

    if args.method == 'dfc':
        print("Start pretraining individual golden standard DECs")
        print("loading the golden standard group 0 encoder")
        encoder_group_0 = get_encoder(args, "encoder_0", args.encoder_0_legacy_path, args.encoder_0_path, [
                                      dataloader_0], device=device, encoder_type=args.encoder_type)
        # if args.encoder_type == 'resnet50':
        #     set_parameter_requires_grad(encoder_group_0,False)

        print("loading the golden standard group 1 encoder")
        encoder_group_1 = get_encoder(args, "encoder_1", args.encoder_1_legacy_path, args.encoder_1_path, [
                                      dataloader_1], device=device, encoder_type=args.encoder_type)
        # if args.encoder_type == 'resnet50':
        #     set_parameter_requires_grad(encoder_group_1,False)

        cluster_centers_0 = None
        cluster_centers_1 = None
        if not args.dfc_0_path:
            # We don't have pretrained decs for both groups -> we have to generate cluster centers
            print("Load group 0 initial cluster definitions")
            cluster_centers_0 = get_cluster_centers(args, encoder_group_0, args.cluster_number, [
                                                    dataloader_0], args.cluster_0_path, device=device, save_name="clusters_0")

            print("Load group 1 initial cluster definitions")
            cluster_centers_1 = get_cluster_centers(args, encoder_group_1, args.cluster_number, [dataloader_1],
                                                    args.cluster_1_path, device=device, save_name="clusters_1")

        print("Train golden standard group 0 DEC")
        # note that the weight of the fairness losses are set to 0, making this a DEC instead of a DFC
        dfc_group_0 = get_dec(args, args.dfc_0_path, [
                              dataloader_0], encoder_group_0, "DEC_G0", device=device, centers=cluster_centers_0)

        print("Train golden standard group 1 DEC")
        # note that the weight of the fairness losses are set to 0, making this a DEC instead of a DFC
        dfc_group_1 = get_dec(args, args.dfc_1_path, [
                              dataloader_1], encoder_group_1, "DEC_G1", device=device, centers=cluster_centers_1)

        print("Load cluster centers for final DFC")
        cluster_centers = get_cluster_centers(args, encoder, args.cluster_number, [dataloader_0, dataloader_1],
                                              args.cluster_path, device=device, save_name="clusters_dfc")

        print("Train final DFC")

        loss_tradeoff = lambda _: (1, 1, 1)
        if args.dfc_tradeoff == 'no_fair':
            loss_tradeoff = lambda _: (0, 1, 1)
        elif args.dfc_tradeoff == 'no_struct':
            loss_tradeoff = lambda _: (1, 0, 1)

        dfc = get_dfc(args, args.dfc_path, [dataloader_0, dataloader_1], encoder, "DFC", encoder_group_0=encoder_group_0,
                      encoder_group_1=encoder_group_1, dfc_group_0=dfc_group_0, dfc_group_1=dfc_group_1, device=device,
                      centers=cluster_centers, get_loss_trade_off=loss_tradeoff)
    elif args.method == 'dec':
        print("Load cluster centers for final DEC")
        cluster_centers = get_cluster_centers(args, encoder, args.cluster_number, [dataloader_0, dataloader_1],
                                              args.cluster_path, device=device, save_name="clusters_dec")

        print("Train final DEC")
        dec = get_dec(args, None, [dataloader_0, dataloader_1],
                      encoder, "DEC", device=device, centers=cluster_centers)
    else:
        raise NotImplementedError

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # encoders
    parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--encoder_legacy_path", type=str)
    parser.add_argument("--encoder_0_legacy_path", type=str)
    parser.add_argument("--encoder_0_path", type=str)
    parser.add_argument("--encoder_1_legacy_path", type=str)
    parser.add_argument("--encoder_1_path", type=str)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_bs", type=int, default=128)
    parser.add_argument("--encoder_max_epochs", type=int, default=50)
    parser.add_argument("--encoder_type", type=str, default='vae')

    # clusters
    parser.add_argument("--cluster_0_path", type=str)
    parser.add_argument("--cluster_1_path", type=str)
    parser.add_argument("--cluster_number", type=int, default=10)
    parser.add_argument("--cluster_path", type=str)
    parser.add_argument("--cluster_n_init", type=int, default=20)
    parser.add_argument("--cluster_max_step", type=int, default=5000)

    # dfc
    parser.add_argument("--dfc_0_path", type=str)
    parser.add_argument("--dfc_1_path", type=str)
    parser.add_argument("--dfc_path", type=str)
    parser.add_argument("--dfc_hidden_dim", type=int, default=64)
    parser.add_argument("--adv_multiplier", type=float, default=10.0)
    parser.add_argument("--dfc_tradeoff", type=str, default='none')

    # dec
    parser.add_argument("--dec_lr", type=float, default=0.001)
    parser.add_argument("--dec_batch_size", type=int, default=512)
    parser.add_argument("--dec_iters", type=int, default=20000)

    # dataset
    parser.add_argument("--dataset", type=str, default="mnist_usps")
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--digital_dataset", type=bool, default=True)

    parser.add_argument("--method", type=str, default="dfc")

    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--test_interval", type=int, default=5000)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default="./DFC_LOGS/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2019)

    parser.add_argument("--half_tensor", default=False, action='store_true')
    args = parser.parse_args()
    wandb.init(project="dfc", entity="fact-dfc", config=args)
    main(args)
