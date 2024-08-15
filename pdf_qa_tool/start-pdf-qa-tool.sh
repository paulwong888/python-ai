#!/bin/bash

# 获取脚本所在目录的绝对路径
script_dir=$(dirname $(readlink -f $0))

echo $script_dir

streamlit run $script_dir/main.py