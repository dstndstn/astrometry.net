#/bin/bash
#
# calling sequence
# ./rc3.sh apikey > ./script
# bash ./script

for file in $(ls ~/rc3/*hard.jpg)
do
    cmd="python ../client.py --server http://staging.astrometry.net/api/ --apikey $1 --upload $file"
    echo $cmd
done
