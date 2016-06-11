from my_settings import *

from simnibs import *

subjects = ["p2", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12",
            "p13"]

for subject in subjects:
    make_symbolic_links(subject, subjects_dir)
    # convert_surfaces(subject, subjects_dir)
    # copy_surfaces(subject, subjects_dir)
    # setup_mne_c_forward(subject)
