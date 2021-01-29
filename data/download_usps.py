import torchvision
import pathlib
import argparse


def download_usps(data_dir):
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    dataset = torchvision.datasets.USPS(root=data_dir, download=True, train=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.Resize(32)]
                                        ))

    # empty the index file
    open("train_usps.txt", 'w').close()

    with open("train_usps.txt", "a") as index_file:
        for idx, (img, label) in enumerate(dataset):
            path = data_dir + '/{:05d}.jpg'.format(idx)
            img.save(path)
            index_file.write(f"{path[1:]} {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # encoders
    parser.add_argument("--data_dir", type=str, default='../data/train_usps/')
    args = parser.parse_args()
    download_usps(args.data_dir)
