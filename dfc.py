import torch
import wandb
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torch.nn import Parameter

from adverserial import AdversarialNetwork, adv_loss
from dataloader import mnist_usps
from eval import predict, cluster_accuracy, balance, tsne_visualization
from utils import set_seed, AverageMeter, aff, target_distribution, inv_lr_scheduler
import matplotlib.pyplot as plt


class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers):
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        Args:
            cluster_number: number of clusters
            embedding_dimension: embedding dimension of feature vectors
            alpha: representing the degrees of freedom in the t-distribution, default 1.0
            cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """

        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number,
                self.embedding_dimension,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch):
        """
            Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
            for each cluster.
        Args:
            batch (FloatTensor): [batch size, embedding dimension]

        Returns:
             FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))

        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DFC(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, alpha=1):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.
        Args:
            cluster_number: number of clusters
            hidden_dimension: hidden dimension, output of the encoder
            alpha: parameter representing the degrees of freedom in the t-distribution, default = 1
        """

        super(DFC, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha, cluster_centers=None)

    def forward(self, batch):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        Args:
            batch (FloatTensor): [batch size, embedding dimension]
        Returns:
            FloatTensor: [batch_size, number of clusters]
        """
        return self.assignment(batch)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


def train(args, dataloader_list, encoder, encoder_group_0=None, encoder_group_1=None, dfc_group_0=None,
          dfc_group_1=None, device='cpu', centers=None, get_loss_trade_off=lambda step: (10, 10, 10), save_name='DFC'):
    """Trains DFC and optionally the critic,

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

    set_seed(args.seed)
    if args.half_tensor:
        torch.set_default_tensor_type('torch.HalfTensor')

    dfc = DFC(cluster_number=args.cluster_number, hidden_dimension=args.dfc_hidden_dim).to(device)
    wandb.watch(dfc)

    critic = AdversarialNetwork(in_feature=args.cluster_number,
                                hidden_size=32,
                                max_iter=args.iters,
                                lr_mult=args.adv_multiplier).to(device)
    wandb.watch(critic)

    if not (centers is None):
        cluster_centers = centers.clone().detach().requires_grad_(True).to(device)
        with torch.no_grad():
            print("loading clustering centers...")
            dfc.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)

    encoder_param = encoder.get_parameters() if args.encoder_type == 'vae' else [
        {"params": encoder.parameters(), "lr_mult": 1}]
    optimizer = torch.optim.Adam(dfc.get_parameters() + encoder_param + critic.get_parameters(), lr=args.dec_lr,
                                 weight_decay=5e-4)

    criterion_c = nn.KLDivLoss(reduction="sum")
    criterion_p = nn.MSELoss(reduction="sum")
    C_LOSS = AverageMeter()
    F_LOSS = AverageMeter()
    P_LOSS = AverageMeter()

    partition_loss_enabled = True
    if not encoder_group_0 or not encoder_group_1 or not dfc_group_0 or not dfc_group_1:
        print("Missing Golden Standard models, switching to DEC mode instead of DFC.")
        partition_loss_enabled = False

    if partition_loss_enabled:
        encoder_group_0.eval(), encoder_group_1.eval()
        dfc_group_0.eval(), dfc_group_1.eval()

    print("Start training")
    assert 0 < len(dataloader_list) < 3
    len_image_0 = len(dataloader_list[0])
    len_image_1 = len(dataloader_list[1]) if len(dataloader_list) == 2 else None
    for step in range(args.iters):
        encoder.train()
        dfc.train()

        if step % len_image_0 == 0:
            iter_image_0 = iter(dataloader_list[0])
        if len_image_1 and step % len_image_1 == 0:
            iter_image_1 = iter(dataloader_list[1])

        image_0, _ = iter_image_0.__next__()
        image_0 = image_0.to(device)
        if not (len_image_1 is None):
            image_1, _ = iter_image_1.__next__()
            image_1 = image_1.to(device)
            image = torch.cat((image_0, image_1), dim=0)
        else:
            image_1 = None
            image = torch.cat((image_0,), dim=0)

        if args.encoder_type == 'vae':
            z, _, _ = encoder(image)
        elif args.encoder_type == 'resnet50':
            z = encoder(image)

        else:
            raise Exception('Wrong encoder type, how did you get this far in running the code?')
        output = dfc(z)
        features_enc_0 = encoder_group_0(image_0)[0] if args.encoder_type == 'vae' else encoder_group_0(image_0)
        predict_0 = dfc_group_0(features_enc_0)
        features_enc_1 = encoder_group_1(image_1)[0] if args.encoder_type == 'vae' else encoder_group_1(image_1)
        predict_1 = dfc_group_1(features_enc_1) if not (image_1 is None) else None

        output_0, output_1 = output[0:args.bs, :], output[args.bs:args.bs * 2, :] if not (predict_1 is None) else None
        target_0, target_1 = target_distribution(output_0).detach(), target_distribution(output_1).detach() if not (
                predict_1 is None) else None

        # Equaition (5) in the paper
        # output_0 and output_1 are probability distribution P of samples being assinged to a class in k
        # target_0 and target_1 are auxiliary distribuion Q calculated based on P. Eqation (4) in the paper
        if not (output_1 is None):
            clustering_loss = 0.5 * criterion_c(output_0.log(), target_0) + 0.5 * criterion_c(output_1.log(), target_1)
        else:
            clustering_loss = criterion_c(output_0.log(), target_0)

        # Equation (2) in the paper
        # output = D(A(F(X)))
        # critic is the distribuition of categorical sensitive subgroup variable G (?)
        if len(dataloader_list) > 1:
            fair_loss, critic_acc = adv_loss(output, critic, device=device)
        else:
            fair_loss, critic_acc = 0, 0

        if partition_loss_enabled:
            # Equation (3) in the paper
            # output_0 and output_1 are the output of the pretrained encoder
            # predict_0 and predict_1 are the soft cluster assignments of the DFC.
            # loss is high if the outputs and predictions (and this the cluster structures) differ.
            if not (predict_1 is None):
                partition_loss = 0.5 * criterion_p(aff(output_0), aff(predict_0).detach()) \
                                 + 0.5 * criterion_p(aff(output_1), aff(predict_1).detach())
            else:
                partition_loss = criterion_p(aff(output_0), aff(predict_0).detach())
        else:
            partition_loss = 0

        loss_trade_off = get_loss_trade_off(step)
        if args.encoder_type == 'resnet50' and args.dataset == 'office_31':  # alpha_s
            loss_trade_off = list(loss_trade_off)
            loss_trade_off[1] = ((512 / 128) ** 2) * (31 / 10)

        total_loss = loss_trade_off[0] * fair_loss + loss_trade_off[1] * partition_loss + loss_trade_off[
            2] * clustering_loss

        optimizer = inv_lr_scheduler(optimizer, args.lr, step, args.iters)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        C_LOSS.update(clustering_loss)
        F_LOSS.update(fair_loss)
        P_LOSS.update(partition_loss)

        wandb.log({f"{save_name} Train C Loss Avg": C_LOSS.avg, f"{save_name} Train F Loss Avg": F_LOSS.avg,
                   f"{save_name} Train P Loss Avg": P_LOSS.avg, f"{save_name} step": step,
                   f"{save_name} Critic ACC": critic_acc})
        wandb.log({f"{save_name} Train C Loss Cur": C_LOSS.val, f"{save_name} Train F Loss Cur": F_LOSS.val,
                   f"{save_name} Train P Loss Cur": P_LOSS.val, f"{save_name} step": step})

        if step % args.test_interval == args.test_interval - 1 or step == 0:

            predicted, labels = predict(dataloader_list, encoder, dfc, device=device, encoder_type=args.encoder_type)
            predicted, labels = predicted.cpu().numpy(), labels.numpy()
            _, accuracy = cluster_accuracy(predicted, labels, args.cluster_number)
            nmi = normalized_mutual_info_score(labels, predicted, average_method="arithmetic")
            bal, en_0, en_1 = balance(predicted, len_image_0, k=args.cluster_number)

            wandb.log(
                {f"{save_name} Train Accuracy": accuracy, f"{save_name} Train NMI": nmi, f"{save_name} Train Bal": bal,
                 f"{save_name} Train Entropy 0": en_0,
                 f"{save_name} Train Entropy 1": en_1, f"{save_name} step": step})

            print("Step:[{:03d}/{:03d}]  "
                  "Acc:{:2.3f};"
                  "NMI:{:1.3f};"
                  "Bal:{:1.3f};"
                  "En:{:1.3f}/{:1.3f};"
                  "Clustering.loss:{C_Loss.avg:3.2f};"
                  "Fairness.loss:{F_Loss.avg:3.2f};"
                  "Partition.loss:{P_Loss.avg:3.2f};".format(step + 1, args.iters, accuracy, nmi, bal, en_0,
                                                             en_1, C_Loss=C_LOSS, F_Loss=F_LOSS, P_Loss=P_LOSS))

            # log tsne visualisation
            if args.encoder_type == "vae":
                tsne_img = tsne_visualization(dataloader_list, encoder, args.cluster_number,
                                              encoder_type=args.encoder_type, device=device)

                if not (tsne_img is None):
                    wandb.log({f"{save_name} TSNE": plt, f"{save_name} step": step})

    torch.save(dfc.state_dict(), f'{args.log_dir}DFC_{save_name}.pth')

    if len(dataloader_list) > 1:
        torch.save(critic.state_dict(), f'{args.log_dir}CRITIC_{save_name}.pth')

    return dfc
