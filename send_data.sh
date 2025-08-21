#!/bin/bash

# 代理服务器信息
PROXY_USER="root"
PROXY_HOST="38.207.167.34"
PROXY_PORT="5000"  # SSH端口，默认为22

# 目标服务器信息
TARGET_USER="yw001"
TARGET_HOST="10.26.0.2"
TARGET_PORT="22"  # 目标服务器的SSH端口

# 要传输的文件或目录
SOURCE="nextqa/extract_1k.ipynb"
DESTINATION="/home/yw001"

echo 6y0hv2X9ZQTbtpj7
echo 980625


# scp -o ProxyCommand="ssh -p $PROXY_PORT $PROXY_USER@$PROXY_HOST -W %h:%p" \
#     $SOURCE $TARGET_USER@$TARGET_HOST:$DESTINATION 

# # 使用ProxyCommand通过SSH代理服务器执行rsync
rsync -avz --progress \
  -e "ssh -o ProxyCommand='ssh -p $PROXY_PORT $PROXY_USER@$PROXY_HOST -W %h:%p' -o KexAlgorithms=diffie-hellman-group-exchange-sha256" \
  $SOURCE $TARGET_USER@$TARGET_HOST:$DESTINATION