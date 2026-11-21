#!/bin/sh

cleanup () {
    if [ $removeOverlay -eq 1 ]; then
        echo "Cleaning up: Removing $distSrc."
        rm "$distSrc"
    fi
    exit
}

removeOverlay=0

if [ $# -lt 4 ]; then
    echo "Usage:"
    echo "  $0 ( <DOT-SIZE> | <DOT-SRC-FILE> ) ( <DOT-AMOUNT> | <DOT-POSITIONS-FILE> ) <SRC-IMG> <DEST-IMG>"
    echo "  If a source file for distractor positions is used, each line must have the following format:"
    echo "  ([+][0-9]+){2}"
    echo "  For example, the line \"+500+200\" draws a distractor at X 500 Y 200."
    exit
fi

if ! [ -f "$3" ]; then
    echo "Source image $3 not found."
    exit
fi

if [ -e "$4" ]; then
    echo "$4 already exists."
    exit
fi

distSrc="$1"

if [ "$1" -eq "$1" ] 2> /dev/null; then
    echo "$1 is integer. Interpreting as size of distractor."
    if [ -f "dot.png" ]; then
        echo "dot.png already exists. Not generating."
        exit
    fi

    convert -size "$1"x"$1" "radial-gradient:white-rgba(255, 255, 255, 0.0)" -depth 8 dot.png

    removeOverlay=1
    distSrc=dot.png
else
    echo "$distSrc is not an integer. Interpreting as image source file for distractor."
    if ! [ -f "$distSrc" ]; then
        echo "Overlay image $distSrc not found."
        exit
    fi
fi

wholeCommand="convert $3"

if [ "$2" -eq "$2" ] 2> /dev/null; then
    echo "$2 is integer. Interpreting as amount of distractors to add."

    xSize=$(identify -format '%w' "$3")
    ySize=$(identify -format '%h' "$3")
    echo "Dimensions of source image are X $xSize Y $ySize."

    for _ in $(seq $2); do
        x=$(( $(od -An -tu4 -N4 /dev/urandom) % $xSize ))
        y=$(( $(od -An -tu4 -N4 /dev/urandom) % $ySize ))
        wholeCommand="$wholeCommand dot.png -geometry +$x+$y -composite"
    done
else
    echo "$2 is not an integer. Interpreting as source file for positions."
    if ! [ -f "$2" ]; then
        echo "$2 not found."
        cleanup
    fi
    while read line; do
        wholeCommand="$wholeCommand dot.png -geometry $line -composite"
    done < "$2"
fi

wholeCommand="$wholeCommand $4" # Add dest file
$wholeCommand

cleanup
