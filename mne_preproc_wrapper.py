# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:50:00 2016

@author: mje
"""

from preprocessing import *

data_folder = '/media/mje/My_Book/Data/agency_connectivity/'


# subjects = ["P2", "P3", "P4", "P5", "P6", "P7", "P8",
#             "P10", "P11", "P12", "P13"]
subjects = ["P13"]


for subject in subjects:
    convert_bdf2fif(subject, data_folder)
    filter_raw(subject, data_folder)

    compute_ica(subject, data_folder)
    save_event_file(subject, data_folder)