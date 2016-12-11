# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:50:00 2016

@author: mje
"""
# from my_settings import (data)
# from preprocessing import (convert_bdf2fif, filter_raw, compute_ica,
#                            save_event_file_ctl)

from preprocessing import save_event_file_ctl

# data_folder = '/media/mje/My_Book/Data/agency_connectivity/'
# data_folder = "/Volumes/My_Passport/agency_connectivity/tmp/"
data_folder = '/Volumes/My_Passport/agency_connectivity/Data/tmp/'

# subjects = ["P2", "P3", "P4", "P5", "P6", "P7", "P8",
#             "P10", "P11", "P12", "P13"]
subjects = ["P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29",
            "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38"]

for subject in subjects:
    # convert_bdf2fif(subject, data_folder)
    # filter_raw(subject, data_folder)

    # compute_ica(subject, data_folder)
    save_event_file_ctl(subject, data_folder)
