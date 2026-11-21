#!/bin/sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <PATH> <FILE-STEM>"
    echo "  Examples:"
    echo "    $0 demo apod2"
    echo "    $0 demo \"*\""
    exit
fi

find "$1" -mindepth 1 -maxdepth 1 \( -name "$2.axy" -o -name "$2.corr" -o -name "$2.match"\
    -o -name "$2.new" -o -name "$2.rdls" -o -name "$2.wcs" -o -name "$2.solved"\
    -o -name "$2-indx.*" -o -name "$2-ngc.*" -o -name "$2-objs.*" \) \
    -exec echo {} \; -exec rm {} \;
