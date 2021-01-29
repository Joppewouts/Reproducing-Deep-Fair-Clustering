import requests
import zipfile
import argparse
import os


# Dataset webpage
# http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html


def rewrite_path_labels(data_dir):
    """  Rewrite path to training data to include "data/MTFL"
    and remove the facial points data we don't use.
    """
    # Just the training data, can do the same for test data by just changing path
    out_name = data_dir + "train_MTFL.txt"
    in_name = data_dir + "MTFL/training.txt"
    # Data for split of glasses/no glasses
    glasses_name = data_dir + "train_MTFL_with_glasses.txt"
    wo_glasses_name = data_dir + "train_MTFL_wo_glasses.txt"  # without
    file_read = open(in_name, "r")
    file_write = open(out_name, "w")

    file_glasses = open(glasses_name, "w")  # Training split based on glasses
    file_wo_glasses = open(wo_glasses_name, "w")

    for line in file_read:
        split_line = line.split()  # Split to avoid data points
        if len(split_line) == 0:  # Catch if last line is empty
            continue
        label = int(split_line[-4]) - 1  # default is 1 and 2 but we need 0 and 1 due to index in lists
        # print(label)
        # filepath gender_label
        # e.g. "./data/MTFL/net_7876\_40_1154_0.jpg 1"

        new_line = data_dir + 'MTFL/' + \
                   split_line[0] + " " + str(label) + '\n'

        new_line = new_line.replace('\\', '/')  # fix default slash from unzip
        # print(new_line)
        if int(split_line[-2]) == 1:  # With glasses training data
            file_glasses.write(new_line)  # write to file
            file_glasses.flush()
        else:  # without glasses training data
            file_wo_glasses.write(new_line)  # write to file
        file_write.write(new_line)  # write all training data path

    file_write.close()
    file_glasses.close()
    file_wo_glasses.close()


def download_mtfl(data_dir):
    if os.getcwd().split('/')[-1] == 'data':
        data_dir = '.' + data_dir
    url = "http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip"

    filename = url.split('/')[-1]  # Should be "MTFL.zip"
    files_in_path = os.listdir(path=data_dir)
    for item in files_in_path:  # If we want to print files in path
        # print(item)
        continue
    if filename not in files_in_path:  # Check if we have downloaded the file
        r = requests.get(url)
        open(data_dir + 'MTFL.zip', 'wb').write(r.content)
    else:
        print(filename, "file exists already... continuing without downloading...")

    with zipfile.ZipFile(data_dir + filename, "r") as zf:  # Extract data in folder
        print("Extracting data from Zip file...")
        zf.extractall(data_dir + "MTFL/")
    print("Finished downloading and unziping MTFL data")
    rewrite_path_labels(data_dir)  # Rewrite the labels so we can use the dataloader
    print("Finished preparing MTFL data")


# NOTE: They test with glasses vs without in cluster for male/female
# LABELS
# --gender: 1 for male, 2 for female
# --smile: 1 for smiling, 2 for not smiling
# --glasses: 1 for wearing glasses, 2 for not wearing glasses.
# --head pose: 1 for left profile, 2 for left, 3 for frontal, 4 for right, 5 for right profile
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # encoders
    parser.add_argument("--data_dir", type=str, default="./data/")
    args = parser.parse_args()

    download_mtfl(args.path_prefix)
