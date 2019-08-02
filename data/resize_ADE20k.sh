#!/bin/bash
# In place resampling to 512 x 1024 px
# Requires imagemagick on a *nix system
# Modify according to your directory structure
var = 0
for f in ./NWPU-RESISC45/NWPU-RESISC45/airport/*.jpg; do
    convert  -resize 1024x512 $f $f
    ((var++))
    printf '%d %s\n'  $var $f
done




