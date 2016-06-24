from __future__ import print_function
import subprocess

# path to submit_to_isis
cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"
proj_code = "MINDLAB2015_MEG-CorticalAlphaAttention"

subjects = ["fs_p3", "fs_p14", "fs_p15", "fs_p16", "fs_p17", "fs_p18", "fs_p19"]

subjects = ["fs_p6"]

subjects_dir = "/home/mje/Dropbox/fs_struct"

for subject in subjects:
    convert_cmd_lh = "mri_surf2surf --srcsubject fsaverage " +  \
                     "--trgsubject %s --hemi lh " % subject + \
                     "--sval-annot %s/fsaverage/label/lh.PALS_B12_Brodmann.annot "  %subjects_dir+ \
                     "--tval %s/%s/label/lh.PALS_B12_Brodmann.annot" % (subjects_dir, subject)
    convert_cmd_rh = "mri_surf2surf --srcsubject fsaverage " +  \
                     "--trgsubject %s --hemi rh " % subject + \
                     "--sval-annot %s/fsaverage/label/rh.PALS_B12_Brodmann.annot " %subjects_dir + \
                     "--tval %s/%s/label/rh.PALS_B12_Brodmann.annot" % (subjects_dir, subject)
         
    subprocess.Popen(convert_cmd_lh,
                     shell=True,
                     stdout=subprocess.PIPE)

    subprocess.Popen(convert_cmd_rh,
                     shell=True,
                     stdout=subprocess.PIPE)
