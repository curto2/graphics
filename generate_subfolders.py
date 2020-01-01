# Curtó and Zarza.
# c@decurto.ch z@dezarza.ch

# Script to generate subfolder structure to read in Tensorflow
# Put Curtó & Zarza into the folder /graphics/samples/
import pandas as ps
import os
import numpy as np
import shutil

dataset = ps.read_csv('labels/c&z.csv') # Load dataset
file_names = list(dataset['Filename'].values)
# Choose label to load from:
# {Age, Ethnicity, Eyes Color, Facial Hair, Gender, Glasses, Hair Color, Hair Covered, Hair Style, Smile, Visible Forehead}
img_labels = list(dataset['Ethnicity'].values) # For instance: Ethnicity 

folders_to_be_created = np.unique(img_labels)

source = os.getcwd()

for new_path in folders_to_be_created:
    if not os.path.exists(new_path):
        os.makedirs(new_path)

folders = folders_to_be_created.copy()

for z in range(len(file_names)):

  current_img = file_names[z]
  current_label = img_labels[z]

  destination_image = shutil.move(source+'/graphics/samples/'+current_img,source+'/'+current_label)
