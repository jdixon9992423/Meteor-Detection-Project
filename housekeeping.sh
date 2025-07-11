#!/bin/sh


#deletes files older than 3 days, even files in subfolders
find /home/raspberry/Documents/images/ -type f -mtime 3 -name "*.png" -delete

#numfiles=$(ls | grep .png | wc -l )


#maxnumfiles=13000
#echo "$numfiles"

#if [ $numfiles -gt $maxnumfiles ]
#then 

#difference=$((numfiles-maxnumfiles))
#ls -ltr | grep .png | awk '{print $9}' | head -n $difference | xargs rm

#fi
