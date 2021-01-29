# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.manifold import TSNE as TSNE
import matplotlib.pyplot as plt

def predict(data_loader, encoder, dfc, device='cpu', encoder_type = 'vae'):
    """

    Args:
        data_loader:
        encoder:
        dfc:
        device:
        encoder_type:

    Returns:
        feature, label
    """
    features = []
    labels = []
    encoder.eval()
    dfc.eval()

    with torch.no_grad():
        for loader in data_loader:
            for idx, (img, label) in enumerate(loader):
                img = img.to(device)
                feat = dfc(encoder(img)[0]) if encoder_type =='vae' else dfc(encoder(img))
                features.append(feat.detach())
                labels.append(label)

    return torch.cat(features).max(1)[1], torch.cat(labels).long()


def tsne_visualization(data_loader, encoder, cluster_number, encoder_type = 'vae', device='cpu', max_batch_per_dataset=3):
    """

    Args:
        data_loader:
        encoder:
        cluster_number:
        encoder_type:
        device:
        max_batch_per_dataset:

    Returns:
        plt
    """

    encoder.eval()

    features = []
    labels = []
    subgroups = []
    try:
        with torch.no_grad():
            for loader_i, loader in enumerate(data_loader):

                for idx, (img, label) in enumerate(loader):
                    if idx >= max_batch_per_dataset:
                        break
                    img = img.to(device)
                    feat = encoder(img)[0] if encoder_type =='vae' else encoder(img)
                    # feat = dfc(encoder(img)[0])
                    # feat = dfc(encoder(img))
                    features.append(feat.detach())
                    labels.append(label)
                    subgroups.append((torch.ones(label.size()) * loader_i))

        features = torch.cat(features).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()
        subgroups = torch.cat(subgroups).detach().cpu().numpy()

        features_embedded = TSNE().fit_transform(features)

        subgroup_markers = ('v', 's', 'o',  'x')

        if cluster_number <= 10:
            cluster_colors = ('tab:brown', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:olive', 'tab:purple', 'tab:pink', 'tab:red', 'tab:gray')
        else:
            cluster_colors = ('tab:orange',)

        ax = plt.figure(figsize=(16, 10))

        # loop over every
        for c in range(cluster_number):
            c_indices = labels == c
            for d in range(len(data_loader)):
                d_indices = subgroups == d
                combined = c_indices & d_indices
                X = features_embedded[combined]
                marker = subgroup_markers[d % len(subgroups)]
                color = cluster_colors[c % len(cluster_colors)]
                plt.scatter(X[:,0], X[:,1], marker=marker, color=color, alpha=0.5)
    except Exception as e:
        print("TSNE failed", str(e))
        return None
    return ax


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    Args:
        y_true (list): list of true cluster numbers, an integer array 0-indexed
        y_predicted (list): list of predicted cluster numbers, an integer array 0-indexed
        cluster_number (int): number of clusters, if None then calculated from entropy_input
    Returns:
        reassignment dictionary, clustering accuracy
    """

    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size

    return reassignment, accuracy


def entropy(entropy_input):
    '''

    Args:
        entropy_input (Tensor):
    Returns:
        entropy (float)
    '''
    epsilon = 1e-5  # for numerical stability
    entropy = -entropy_input * torch.log(entropy_input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def balance(predicted, size_0, k=10):
    """

    Args:
        predicted:
        size_0:
        k:

    Returns:

    """
    count = torch.zeros((k, 2), dtype=float)
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5
    
    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])

    en_0 = entropy(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()
