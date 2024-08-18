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


# # 使用 find 查找以 ASCC19 开头的文件
# find "$search_dir" -type f -name "ASCC19*" | while read file; do
#   # 构造新的文件名，将 ASCC19 替换为 ASCC_19
#   new_file=$(echo "$file" | sed 's/ASCC19/ASCC_19/')
  
#   # 重命名文件
#   mv "$file" "$new_file"
# done
