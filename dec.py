import torch
import wandb
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torch.nn import Parameter

from adverserial import AdversarialNetwork, adv_loss
from dataloader import mnist_usps
from dfc import DFC
from eval import predict, cluster_accuracy, balance, tsne_visualization
from utils import set_seed, AverageMeter, aff, target_distribution, inv_lr_scheduler
import matplotlib.pyplot as plt
from resnet50_finetune import *


def train(args, dataloader_list, encoder, device='cpu', centers=None, save_name='DEC'):
    """
        Trains DFC and optionally the critic,
        automatically saves when finished training

    Args:
        args: Namespace object which contains config set from argument parser
              {
                lr,
                seed,
                iters,
                log_dir,
                test_interval,
                adv_multiplier,
                dfc_hidden_dim
              }
        dataloader_list (list): this list may consist of only 1 dataloader or multiple
        encoder: Encoder to use
        encoder_group_0: Optional pre-trained golden standard model
        encoder_group_1: Optional pre-trained golden standard model
        dfc_group_0: Optional cluster centers file obtained with encoder_group_0
        dfc_group_1: Optional cluster centers file obtained with encoder_group_1
        device: Device configuration
        centers: Initial centers clusters if available
        get_loss_trade_off: Proportional importance of individual loss functions
        save_name: Prefix for save files

    Returns:
        DFC: A trained DFC model

    """
    # """
    # Function for training and testing a VAE model.
    # Inputs:
    #     args -
    # """

    set_seed(args.seed)

    if args.half_tensor:
        torch.set_default_tensor_type('torch.HalfTensor')

    dec = DFC(cluster_number=args.cluster_number, hidden_dimension=args.dfc_hidden_dim).to(device)
    wandb.watch(dec)

    if not (centers is None):
        cluster_centers = centers.clone().detach().requires_grad_(True).to(device)
        with torch.no_grad():
            print("loading clustering centers...")
            dec.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)
    # depending on the encoder we get the params diff so we have to use this if
    encoder_param = encoder.get_parameters() if args.encoder_type == 'vae' else [
        {"params": get_update_param(encoder), "lr_mult": 1}]
    optimizer = torch.optim.Adam(dec.get_parameters() + encoder_param, lr=args.dec_lr)

    # criterion_c = nn.KLDivLoss(reduction="sum")
    # following dec code more closely
    criterion_c = nn.KLDivLoss(size_average=False)

    C_LOSS = AverageMeter()

    print("Start training")
    assert 0 < len(dataloader_list) < 3

    concat_dataset = torch.utils.data.ConcatDataset([x.dataset for x in dataloader_list])
    training_dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        batch_size=args.dec_batch_size,
        shuffle=True,
        num_workers=4
    )

    for step in range(args.dec_iters):
        encoder.train()
        dec.train()

        if step % len(training_dataloader) == 0:
            iterator = iter(training_dataloader)

        image, _ = iterator.__next__()
        image = image.to(device)
        if args.encoder_type == 'vae':
            z, _, _ = encoder(image)
        elif args.encoder_type == 'resnet50':
            z = encoder(image)
        else:
            raise Exception('Wrong encoder type, how did you get this far in running the code?')
        output = dec(z)

        target = target_distribution(output).detach()

        clustering_loss = criterion_c(output.log(), target) / output.shape[0]

        optimizer.zero_grad()
        clustering_loss.backward()
        optimizer.step()

        C_LOSS.update(clustering_loss)

        wandb.log({f"{save_name} Train C Loss Avg": C_LOSS.avg, f"{save_name} step": step})
        wandb.log({f"{save_name} Train C Loss Cur": C_LOSS.val, f"{save_name} step": step})

        if step % args.test_interval == args.test_interval - 1 or step == 0:
            predicted, labels = predict(dataloader_list, encoder, dec, device=device, encoder_type=args.encoder_type)
            predicted, labels = predicted.cpu().numpy(), labels.numpy()
            _, accuracy = cluster_accuracy(predicted, labels, args.cluster_number)
            nmi = normalized_mutual_info_score(labels, predicted, average_method="arithmetic")
            bal, en_0, en_1 = balance(predicted, len(dataloader_list[0]), k=args.cluster_number)

            wandb.log(
                {f"{save_name} Train Accuracy": accuracy, f"{save_name} Train NMI": nmi, f"{save_name} Train Bal": bal,
                 f"{save_name} Train Entropy 0": en_0,
                 f"{save_name} Train Entropy 1": en_1, f"{save_name} step": step})

            print("Step:[{:03d}/{:03d}]  "
                  "Acc:{:2.3f};"
                  "NMI:{:1.3f};"
                  "Bal:{:1.3f};"
                  "En:{:1.3f}/{:1.3f};"
                  "Clustering.loss:{C_Loss.avg:3.2f};".format(step + 1, args.dec_iters, accuracy, nmi, bal, en_0,
                                                              en_1, C_Loss=C_LOSS))

            # log tsne visualisation
            if args.encoder_type == "vae":
                tsne_img = tsne_visualization(dataloader_list, encoder, args.cluster_number,
                                              encoder_type=args.encoder_type,
                                              device=device)
                if not (tsne_img is None):
                    wandb.log({f"{save_name} TSNE": plt, f"{save_name} step": step})

    torch.save(dec.state_dict(), f'{args.log_dir}DFC_{save_name}.pth')

    return dec
