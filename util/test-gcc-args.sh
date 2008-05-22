#! /bin/bash

CC="$1"
TEST_OPTION="$2"
TEST_FAIL_OPTION="$3"
OP="$4"

TMPFILE=`mktemp -t whatever.XXXXXX`

(${CC} ${TEST_OPTION} ${OP} -o ${TMPFILE} -x c - \
    < /dev/null > /dev/null 2>/dev/null \
    && echo "${TEST_OPTION}" ) \
    || echo "${TEST_FAIL_OPTION}"

rm ${TMPFILE}
