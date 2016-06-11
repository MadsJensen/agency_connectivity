from __future__ import print_function
import os
import sys
import subprocess
from my_settings import *


def make_symbolic_links(subject, subjects_dir):
    """Make symblic links between FS dir and subjects_dir.

    Parameters
    ----------
    fname : string
        The name of the subject to create for
    subjects_dir : string
        The subjects dir for FreeSurfer
    """

    os.chdir(subjects_dir + subject)
    sp = subprocess.Popen("ln -s fs_%s/* ." % subject,
                          shell=True,
                          stdout=subprocess.PIPE)
    sp.communicate()[0]


def convert_surfaces(subject, subjects_dir):
    """Convert the SimNIBS surface to FreeSurfer surfaces.

    Parameters
    ----------
    subject : string
       The name of the subject
    subjects_dir : string
        The subjects dir for FreeSurfer
    """
    os.chdir(subjects_dir + subject + "/m2m_%s" % subject)

    convert_csf = subprocess.Popen("meshfix csf.stl -u 10" +
                                   "--vertices 4098 --fsmesh",
                                   shell=True,
                                   stdout=subprocess.PIPE)
    convert_csf.communicate()[0]

    convert_skull = subprocess.Popen("meshfix skull.stl -u 10" +
                                     "--vertices 4098 --fsmesh",
                                     shell=True,
                                     stdout=subprocess.PIPE)
    convert_skull.communicate()[0]

    convert_skin = subprocess.Popen("meshfix skin.stl -u 10" +
                                    "--vertices 4098 --fsmesh",
                                    shell=True,
                                    stdout=subprocess.PIPE)
    convert_skin.communicate()[0]


def copy_surfaces(subject, subjects_dir):
    """Copy the converted FreeSurfer surfaces to the bem dir.

    Parameters
    ----------
    subject : string
       The name of the subject
    subjects_dir : string
        The subjects dir for FreeSurfer
    """
    os.chdir(subjects_dir + subject + "/m2m_%s" % subject)
    copy_inner_skull = "cp -f csf_fixed.fsmesh " + subjects_dir + \
                       "/%s/bem/inner_skull.surf" % subject
    copy_outer_skull = "cp -f skull_fixed.fsmesh " + subjects_dir + \
                       "/%s/bem/outer_skull.surf" % subject
    copy_outer_skin = "cp -f skin_fixed.fsmesh " + subjects_dir + \
                      "/%s/bem/outer_skin.surf" % subject

    convert_skin_to_head = "mne_surf2bem --surf outer_skin.surf" +\
                           "--fif %s-head.fif --id 4" % subject
                           
    sp_inner_skull = subprocess.Popen(copy_inner_skull,
                                      shell=True,
                                      stdout=subprocess.PIPE)

    sp_outer_skull = subprocess.Popen(copy_outer_skull,
                                      shell=True,
                                      stdout=subprocess.PIPE)

    sp_outer_skin = subprocess.Popen(copy_outer_skin,
                                     shell=True,
                                     stdout=subprocess.PIPE)

    sp_inner_skull.communicate()[0]
    sp_outer_skull.communicate()[0]
    sp_outer_skin.communicate()[0]

    os.chdir(subjects_dir + subject + "/bem")

    sp_convert_skin2head = subprocess.Popen(convert_skin_to_head,
                                            shell=True,
                                            stdout=subprocess.PIPE)
    sp_convert_skin2head.communicate()[0]


def setup_mne_c_forward(subject):
    setup_forward = "mne_setup_forward_model --subject " +\
                    "%s --surf --ico -6" % subject

    sp_setup_forward = subprocess.Popen(setup_forward,
                                        shell=True,
                                        stdout=subprocess.PIPE)
    sp_setup_forward.communicate()[0]
