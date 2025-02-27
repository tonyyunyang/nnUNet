#!/bin/bash

# Push subtrees to their respective repositories
echo "Pushing dynamic_architecture to dynamic-network-architectures repo..."
git subtree push --prefix=dynamic_architecture git@github.com:tonyyunyang/dynamic-network-architectures.git main

echo "Pushing nnunetv2 to nnUNet repo..."
git subtree push --prefix=nnunetv2 git@github.com:tonyyunyang/nnUNet.git master

echo "All subtrees pushed successfully!"