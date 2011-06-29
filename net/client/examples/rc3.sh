#/bin/bash
#
# calling sequence
# ./rc3.sh apikey

for file in $(ls ~/rc3/hard*.jpg)
do
    cmd = "python client.py --server http://supernova.astrometry.net/api/ --apikey $1 --upload $file"
done
