#! /bin/bash

INFO=$(svn info)
#echo "info is $INFO"

URL=$(echo "${INFO}" | awk '/^URL:/{ print $2 }')
#echo "url is $URL"

REV=$(echo "${INFO}" | awk '/^Revision:/{ print $2 }')
#echo "rev is $REV"

DATE=$(echo "${INFO}" | awk 'BEGIN{FS=": "} /^Last Changed Date:/{ print $2 }')
#echo "date is $DATE"

echo "static const char* url  = \"${URL}\";"
echo "static const int   rev  = ${REV};"
echo "static const char* date = \"${DATE}\";"

