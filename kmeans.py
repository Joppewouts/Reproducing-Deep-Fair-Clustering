import torch
import wandb
from sklearn.cluster import KMeans
from eval import cluster_accuracy
import numpy as np
from tqdm import tqdm
import os
# Used for test only in main
import argparse
from dataloader import mnist_usps
from module import Encoder


def get_cluster_centers(args=None, autoencoder=None, cluster_number=2, dataloader_list=None,
                        file_path=None, save_name=None, device='cpu'):
    """
    If a file exists with cluster centers we load this file and return it in tensor form
    otherwise we find them using KMeans.
    Args:
        autoencoder (Encoder) -  Encoder used to generate the features to cluster.
        cluster_number (int) - Amount of cluster centers to generate.
        dataloader_list (DataLoader) -  Data to cluster.
        file_path (String) - File path used to retrieve cluster centers.
        save_name (String) - If given, the generated clusters will be saved with that name, otherwise they won't be saved
        device (String)- Device used by autoencoder.
        args.seed (int) - Seed used in KMeans for reproducibility purposes.
        args.log_dir (String) - Directory to save the cluster centers.
        args.cluster_n_init (int)
        args.cluster_max_step (int)
        args.encoder_bs (int)
    Returns:
        cluster_centers (tensor) - The kmeans centers on arg.device with shape (n_clusters, n_features).
    """

    if file_path:  # Load centers from file and return them on device
        print("Loading pretrained KMeans centroids")
        centers = np.loadtxt(file_path)
        cluster_centers = torch.tensor(
            centers, dtype=torch.float, requires_grad=True).to(device)
    else:  # Train Kmeans and generate centers
        # https://github.com/vlukiyanov/pt-dec/blob/11b30553858c1c146a5ee0b696c768ab5244f0ff/ptdec/model.py#L74-L92
        print("Training KMeans for centroids")
        kmeans = KMeans(n_clusters=cluster_number,
                        n_init=args.cluster_n_init, random_state=args.seed, max_iter=args.cluster_max_step)
        autoencoder.eval()
        features = []
        actual = []

        # merge dataloaders
        concat_dataset = torch.utils.data.ConcatDataset([x.dataset for x in dataloader_list])

        dataloader = torch.utils.data.DataLoader(
            dataset=concat_dataset,
            batch_size=args.encoder_bs
        )

        # form initial cluster centres
        data_iterator = tqdm(dataloader,
                             leave=True,
                             unit="batch",
                             disable=False,
                             )
        print("Generating features for kmeans")

        with torch.no_grad():
            # Loop through data and generate features from the encoder.                    
            for index, batch in enumerate(data_iterator):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    # if we have a prediction label, separate it to actual
                    batch, value = batch
                    actual.append(value)
                # Assuming we use the encoder from module.py
                if args.encoder_type == 'vae':
                    feature = autoencoder(batch.to(device))
                elif args.encoder_type == 'resnet50':
                    feature = list()
                    z = autoencoder(batch.to(device))  # [:,:args.dfc_hidden_dim]

                    feature.append(z)

                features.append(feature[0].detach().cpu())
        print("Training samples:", len(features))

        actual = torch.cat(actual).long()  # Save labels as long in torch tensor.
        samples = torch.cat(features)
        print(f"Data shape {samples.shape}")
        print(f"Labels shape {actual.shape}")
        print("Training...")
        predicted = kmeans.fit_predict(samples.numpy(), actual)  # predict centers from features.
        _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())  # Compute accuracy of predictions
        cluster_centers = kmeans.cluster_centers_  # define centers

        if save_name:  # If param. save_name then save the centers.
            filepath = args.log_dir + save_name + ".txt"
            if not os.path.exists(args.log_dir):
                os.mkdir(args.log_dir)
            print("Saving clusters to:", filepath)
            np.savetxt(filepath, cluster_centers)
            if not (wandb.run is None):  # check if wandb is running
                wandb.run.summary[f"{save_name}_accuracy"] = accuracy

        cluster_centers = torch.tensor(  # Convert centers to tensor and send to device.
            cluster_centers, dtype=torch.float, requires_grad=True
        ).to(device)
        print(f"Training KMeans completed, accuracy: {accuracy:.2f}")
    return cluster_centers


# Code to test the functionality of get_cluster_center()
if __name__ == "__main__":
    print('\n \nTesting loading centers...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--centers", type=str, default="save/centers.txt")
    parser.add_argument("--log_dir", type=str, default="clusters_temp/")
    parser.add_argument("--encoder_bs", type=int, default=128)

    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--cluster_n_init", type=int, default=20)
    parser.add_argument("--cluster_max_step", type=int, default=5000)
    parser.add_argument("--half_tensor", type=bool, default=False)
    parser.add_argument("--encoder_type", type=str, default='vae')
    args = parser.parse_args()
    device_local = 'cpu'  # set device for this script
    clusters = get_cluster_centers(file_path=args.centers, device=device_local)  # Test loading centers.txt
    print("Loaded centers with shape", clusters.shape, "\n \n")

    print("Testing generating centers...")

    encoder = Encoder().to(device_local)  # load encoder class
    legacy_path = 'save/encoder_pretrain.pth'  # path to pretrained encoder

    encoder.load_state_dict(torch.load(legacy_path, map_location=device_local))  # load encoder
    data_itt = mnist_usps(args)[0]  # get dataiterator from data loader
    clusters = get_cluster_centers(args=args, autoencoder=encoder, device=device_local,
                                   cluster_number=10, dataloader_list=[data_itt], save_name="test_cluster")
    print("Loaded centers with shape", clusters.shape)
