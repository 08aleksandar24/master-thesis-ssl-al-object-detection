import os
import shutil
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("torchgeo/million-aid", cache_dir="/storage/local/ssd")
dataset.save_to_disk('/storage/local/ssd/MilAID')
