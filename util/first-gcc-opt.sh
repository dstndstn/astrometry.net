#! /bin/bash

CMD="$1"
shift

for x in $*; do
    $CMD $x \
        > /dev/null 2> /dev/null < /dev/null \
        && echo $x && exit 0
done
echo ""
