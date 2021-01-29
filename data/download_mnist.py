import torchvision
import pathlib
import argparse


def download_mnist(data_dir):
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    dataset = torchvision.datasets.MNIST(root=data_dir, download=True, train=True,
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.Resize(32)]
                                         ))

    # empty the index file
    open("train_mnist.txt", 'w').close()

    with open("train_mnist.txt", "a") as index_file:
        for idx, (img, label) in enumerate(dataset):
            path = data_dir + '/{:05d}.jpg'.format(idx)
            img.save(path)
            index_file.write(f"{path[1:]} {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='../data/train_mnist/')
    args = parser.parse_args()
    download_mnist(args.path_prefix)
