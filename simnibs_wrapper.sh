#!/bin/bash


script_dir=/projects/MINDLAB2015_MR-YoungAddiction/scripts/ya_project/mr_analyses

data_dir=/home/mje/Projects/agency_connectivity/Data/T1/
# data_dir="/home/mje/Projects/agency_connectivity/Data/T1/p"

for j in $(seq 5 7);
do
    echo "**************"
    echo "Working on:" "P"$j
    echo "**************"

    export SUBJECTS_DIR=$data_dir"p"$j

    echo $SUBJECTS_DIR

    mri2mesh --brain --subcort --head "p"$j $data_dir/"P"$j/t1.nii
    
    # $script_dir/recon_subject.sh $file
done
