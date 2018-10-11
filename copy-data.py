"""
Serialize the classifier object and trained model
"""

import sys
import os
import gzip
import shutil
import ntpath

def collect_datasets():
    path = "penn-ml-benchmarks-master/datasets/regression"
    collected_path = "data"
    gz_files = list()
    ctr = 1
    for item in os.listdir(path):
        print(ctr)
        sub_dir = path + '/' + item
        if os.path.isdir(sub_dir):
            src_files = os.listdir(path + '/' + item)
            for k in src_files:
                if "tsv" in k:
                    gz_file_path = sub_dir + '/' + k 
                    print(gz_file_path)
                    shutil.copy(gz_file_path, collected_path)
                    ctr += 1
        
collect_datasets()





