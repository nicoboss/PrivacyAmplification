#!/bin/bash
if [ ! -f ~/.ssh/PrivacyAmplificationDocker.pub ]; then
  ssh-keygen -t rsa -b 4096 -f ~/.ssh/PrivacyAmplificationDocker -N "" -C "PrivacyAmplificationDocker"
  ssh-add ~/.ssh/PrivacyAmplificationDocker
  cp -a ~/.ssh/PrivacyAmplificationDocker PrivacyAmplificationDocker.key
  cat PrivacyAmplificationDocker.key | docker run -i cubeearth/puttygen@sha256:42107acb9d1d4a7ce0d6b5d89e20f55c00dcfb76537aa21c5eef621c6905abff > PrivacyAmplificationDocker.ppk
fi
cp -a ~/.ssh/PrivacyAmplificationDocker.pub PrivacyAmplificationDocker.pub
docker build -t privacyamplification -f Dockerfile .
rm -f PrivacyAmplificationDocker.pub
docker run -it --gpus all -p 2222:22 privacyamplification
