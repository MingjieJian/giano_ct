#!/bin/bash

search_dir="../SPA_spectra/2GIANO/ASCC_11/"

find "$search_dir" -type f -name "*.fits" | while read -r file; do
# 使用 find 命令查找后缀为 .fits 的文件，然后用 grep 进行字符串匹配
    echo $file
    output_folder=$(echo "$file" | sed 's/_\([^_]*\)\.fits$//')/
    echo $output_folder
    sh ./giano_con_tell.sh $file $output_folder

    sleep 1

done