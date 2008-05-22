#! /bin/bash

# MacOSX "thinks different" about shared libraries...
TMPFILE=gcc-out.tmp
X=$(first-gcc-opt.sh \
    "${CC} -x c -o ${TMPFILE} -" \
    "-shared" \
    "-dynamic -dynamiclib")
echo "SHAREDLIBFLAGS_DEF := $X"

X=$(first-gcc-opt.sh \
    "${CC} -x c -c -o ${TMPFILE} -" \
    "-ffinite-math-only -fno-signaling-nans")
echo "FLAGS_DEF += $X"

DEFS=$(${CC} -dM -E - < /dev/null)

# use "native" if it's available (gcc 4.2 and above)
X=$(first-gcc-opt.sh \
    "${CC} -x c -c -o ${TMPFILE} -" \
    "-march=native")
if [ "x${X}" != "x" ]; then
    echo "FLAGS_DEF += ${X}";

else
    MACHINE=`uname -m`

    if [ "${MACHINE}" = "i686" ]; then

        # gcc before version 3.1 doesn't support "pentium4";
        #     use "i686" instead.
        
        # Note that we're not really justified in assuming this is a P4...

        X=`first-gcc-opt.sh \
            "${CC} -x c -c -o ${TMPFILE} -" \
            "-march=pentium4" \
            "-march=i686"`
        echo "FLAGS_DEF += $X";

    elif [ "${MACHINE}" = "x86_64" ]; then

        K8=$(echo ${DEFS} | grep "\#define __tune_k8__ 1")

        if [ "x${K8}" != "x" ]; then
            echo "FLAGS_DEF += -march=k8 -m64"
        fi;

    else
        echo "FLAGS_DEF += -DNOT_686";
    fi
fi

APPLE=$(echo ${DEFS} | grep "__APPLE__")
if [ "x${APPLE}" != "x" ]; then
    echo "FLAGS_DEF += -DNOBOOL"
fi

X=$(first-gcc-opt.sh \
    "${CC} -x c -c -o ${TMPFILE} -" \
    "-fno-math-errno -fno-trapping-math")
echo "FLAGS_DEF += $X"

X=$(first-gcc-opt.sh \
    "${CC} -x c -c -o ${TMPFILE} -" \
    "-fno-stack-protector")
echo "FLAGS_DEF += $X"

