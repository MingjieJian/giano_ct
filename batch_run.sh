#!/bin/bash

# # 设定根目录
# root_dir="/srv/scratch/miji0984/SPA_data/giano_spctra/"

# # 遍历根目录下的所有子文件夹
# find "$root_dir" -type d | while read -r folder; do
#     echo "Processing folder: $folder"

#     # 在每个子文件夹中查找 .fits 文件并执行相应操作
#     find "$folder" -type f -name "*.fits" | while read -r file; do
#         echo "Processing file: $file"
        
#         # 输出文件夹的命名规则（和原脚本一样）
#         output_folder=$(echo "$file" | sed 's/_\([^_]*\)\.fits$//')/
#         echo "Output folder: $output_folder"
        
#         # 执行 giano_con_tell.sh 脚本
#         sh ./giano_con_tell.sh "$file" "$output_folder"
        
#         # 暂停 1 秒
#         sleep 1
#     done
# done

search_dir="/srv/scratch/miji0984/SPA_data/giano_spctra/Alessi_Teutsch/"

find "$search_dir" -type f -name "*.fits" | while read -r file; do
# 使用 find 命令查找后缀为 .fits 的文件，然后用 grep 进行字符串匹配
    echo $file
    output_folder=$(echo "$file" | sed 's/_\([^_]*\)\.fits$//')/
    echo $output_folder
    sh ./giano_con_tell.sh $file $output_folder
    sleep 1

done