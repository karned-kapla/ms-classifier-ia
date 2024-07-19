#!/bin/zsh
echo 'version ?'
read version

docker build -t killiankopp/ms-classifier-ia:$version .
docker push killiankopp/ms-classifier-ia:$version

docker tag killiankopp/ms-classifier-ia:$version killiankopp/ms-classifier-ia:latest
docker push killiankopp/ms-classifier-ia:latest
