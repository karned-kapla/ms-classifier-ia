#!/bin/zsh
echo 'version ?'
read version

docker build -t killiankopp/ms-classifier-ia:$version .