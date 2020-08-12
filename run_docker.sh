#!/bin/bash
if [ ! -f ~/.ssh/PrivacyAmplificationDocker.pub ]; then
ssh-keygen -t rsa -b 4096 -f ~/.ssh/PrivacyAmplificationDocker -N "" -C "PrivacyAmplificationDocker"
ssh-add ~/.ssh/PrivacyAmplificationDocker
fi
cp -a ~/.ssh/PrivacyAmplificationDocker.pub PrivacyAmplificationDocker.pub
docker build -t privacyamplification -f Dockerfile .
rm -f PrivacyAmplificationDocker.pub
docker run -it --gpus all -p 2222:22 privacyamplification
