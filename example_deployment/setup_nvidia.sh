#!/bin/bash

DRIVER_VERSION="525"

install_docker() 
{
    apt-get update
    # install essential packages
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
    # add GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    apt-key fingerprint 0EBFCD88
    sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs)\
    stable"

    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
}

install_nvidia_docker_toolkit() 
{
    # install nvidia driver
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    systemctl restart docker
}

install_nvidia_drivers() 
{
    sudo apt autoremove
    apt install software-properties-common -y
    add-apt-repository ppa:graphics-drivers/ppa -y
    sudo apt update
    # install nvidia driver
    ubuntu-drivers install --gpgpu nvidia:$DRIVER_VERSION-server -y
}

install_nvidia_docker_toolkit