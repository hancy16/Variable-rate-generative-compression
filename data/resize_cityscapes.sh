#!/bin/bash
# Author: Grace Han
# In place resampling to 512 x 1024 px
# Requires imagemagick on a *nix system
# Modify according to your directory structure
var = 0
for f in ./leftImg8bit/test/leverkusen/*.png; do
    convert  -resize 1024x512 $f $f
    ((var++))
    printf '%d\n'  $var
done
