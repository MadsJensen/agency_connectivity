#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 21:59:26 2017

@author: au194693
"""
import mne 
import os
from surfer import Brain

subjects_dir = "/Users/au194693/mne_data/MNE-sample-data/subjects"
os.environ["SUBJECTS_DIR"] = subjects_dir
labels = mne.read_labels_from_annot(
    "fsaverage", parc='PALS_B12_Brodmann', regexp="Bro",
    subjects_dir=subjects_dir)

# make combinations of label indices
label_index = [52, 53, 68, 69]

labels_list = [ labels[58] + labels[60], labels[59] + labels[61],
               labels[54] + labels[0], labels[55] + labels[1],
               labels[52], labels[53],
               labels[68], labels[69]]
    

subject_id = "fsaverage"
hemi = "lh"
surf = "inflated"
left_b = Brain(subject_id, hemi, surf)

# If the label lives in the normal place in the subjects directory,
# you can plot it by just using the name

for lbl in labels_list[::2]:
    left_b.add_label(lbl, color="#2E2E2E")
    left_b.add_label(lbl, borders=True, color="#F2F2F2")

left_b.save_single_image("left_hemi.png")

subject_id = "fsaverage"
hemi = "rh"
surf = "inflated"
right_b = Brain(subject_id, hemi, surf)

# If the label lives in the normal place in the subjects directory,
# you can plot it by just using the name

for lbl in labels_list[1::2]:
    right_b.add_label(lbl, color="#2E2E2E")
    right_b.add_label(lbl, borders=True, color="#F2F2F2")

right_b.save_single_image("right_hemi.png")