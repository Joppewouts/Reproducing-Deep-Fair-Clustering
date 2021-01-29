import argparse
from office31 import download_and_extract_office31
from pathlib import Path
import os


# The lib creates the dataset using dataset ops lib
# https://github.com/LukasHedegaard/datasetops
# There is benchmark https://paperswithcode.com/sota/domain-adaptation-on-office-31


def download_office31(data_dir):
    if os.getcwd().split('/')[-1] == 'data':
        data_dir = '.' + data_dir
    data_dir = Path(data_dir)
    download_and_extract_office31(data_dir)
    write_path_data(data_dir.absolute(), "amazon")
    write_path_data(data_dir.absolute(), "webcam")
    print(f"Finished writing paths in {data_dir.absolute()}")


def write_path_data(data_dir, src="amazon"):
    out_name = f"{data_dir}/train_office31_" + src + ".txt"
    file_open = open(out_name, "w")
    label = 0
    d = f'{data_dir}/{src}/images/'
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        for f in os.listdir(full_path):
            f_full = os.path.join(full_path, f)
            if os.path.isfile(f_full):
                txt = f_full + " " + str(label) + '\n'
                file_open.write(txt)
        label += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # encoders
    parser.add_argument("--data_dir", type=str, default="./data/office31/")
    args = parser.parse_args()
    download_office31(args.path_prefix)
