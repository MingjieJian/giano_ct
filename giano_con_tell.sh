#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <spectra_txt_file> <output_folder>"
fi

mkdir -p "$2/"

python giano_con_tell.py "$1" "$2/" > "$2/giano_ct.log" 2>&1