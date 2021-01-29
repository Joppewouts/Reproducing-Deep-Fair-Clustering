from office31 import office31
from office31 import download_and_extract_office31
from pathlib import Path
import os
#Hacky script to count how many images are in each folder/cluster in both sources
out_name = "./data/office31/office31_count.txt"

def count_items(src="amazon"):
    file_open=open(out_name, "a")
    label = 0
    d = "./data/office31/"+src+"/images/"
    count = {}
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        for f in os.listdir(full_path):
            if count.keys().__contains__(path):
                count[path] += 1
            else:
                count[path] = 1
        label +=1
    file_open.write('\n'+src+'\n')
    file_open.write(str(count)+'')

file_open=open(out_name, "w")
file_open.write('')   
count_items("amazon")
count_items("webcam")    