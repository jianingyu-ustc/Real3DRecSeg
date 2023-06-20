#!bin/sh
for file in ../scannet/*
do
    sh construct.sh $file/
done