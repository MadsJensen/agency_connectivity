# data_path = '/home/mje/Projects/agency_connectivity/Data'
# data_path = '/media/mje/My_Book/Data/agency_connectivity/'
data_path = "/Volumes/My_Passport/agency_connectivity/agency_connect_2/"

subjects_dir = data_path + "fs_subjects_dir/"
save_folder = data_path + "filter_ica_data/"
maxfiltered_folder = data_path + "maxfiltered_data/"
epochs_folder = data_path + "epoched_data/"
tf_folder = data_path + "tf_data/"
mne_folder = data_path + "minimum_norm/"
log_folder = data_path + "log_files/"
graph_data = data_path + "graph_data/"

results_folder = "/Volumes/My_Passport/agency_connectivity/results/"

# subjects = ["p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11",
#            "p12", "p13", "P14", "P15", "P16", "P17", "P18", "P19"]
subjects = ["p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11",
            "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19"]

bands = {'delta': [2, 4],
         'theta': [5, 7],
         'alpha': [8, 12],
         'beta': [15, 29],
         'gamma1': [30, 59],
         'gamma2': [60, 90]}
