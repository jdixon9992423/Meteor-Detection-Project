#!/bin/sh


while true
do
    #sshpass -p 'allskycam' rsync -avz raspberry@raspberrypi.local:/home/raspberry/Documents/images ~/Documents/UWI_2023_2024_semester2/research_project
    sshpass -p 'password' rsync -avz raspberry@10.0.1.144:/home/raspberry/Documents/images 'path to save to on server'
    sleep 300
done

