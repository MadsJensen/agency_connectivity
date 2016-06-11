from __future__ import print_function
import subprocess

# path to submit_to_isis
cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"
proj_code = "MINDLAB2015_MEG-CorticalAlphaAttention"

subjects = ["p2", "p4", "p5", "p7", "p8", "p9", "p10", "p11", "p12",
            "p13"]

subjects_dir = "/media/mje/My_Book/Data/agency_connectivity/fs_subjects_dir"

for subject in subjects[:1]:
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
