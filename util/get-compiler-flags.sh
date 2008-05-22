#! /bin/bash

# MacOSX "thinks different" about shared libraries...
TMPFILE=gcc-out.tmp
X=`./first-gcc-opt.sh \
    "$(CC) -x c -o ${TMPFILE} -" \
    "-shared" \
    "-dynamic -dynamiclib"`
echo "SHAREDLIBFLAGS_DEF := $X"

X=`./first-gcc-opt.sh \
    "$(CC) -x c -c -o ${TMPFILE} -" \
    "-ffinite-math-only -fno-signaling-nans"`
echo "FLAGS_DEF += $X"

# use "native" if it's available (gcc 4.2 and above)
NATIVE="-mtune=native -march=native"
if `$(CC) -x c -c -o ${TMPFILE} \
    ${NATIVE} \
    - < /dev/null > /dev/null 2> /dev/null`; then
    echo "FLAGS_DEF += ${NATIVE}"
else
    MACHINE=`uname -m`

    if [ "${MACHINE}" = "i686" ]; then
    # gcc before version 3.1 doesn't support "pentium4"; use "i686" instead.
    # gcc 3.1: "pentium4"

        X=`./first-gcc-opt.sh \
            "$(CC) -x c -c -o ${TMPFILE} -" \
            "-march=pentium4" \
            "-march=i686"`
        echo "FLAGS_DEF += $X"
    elif [ "${MACHINE}" = "x86_64" ]; then
    else
    fi

fi

eq ($(MACHINE), i686)
else
	K8 := $(shell $(CC) -dM -E - < /dev/null | grep "\#define __tune_k8__ 1")
	ifneq ($(K8),)
		FLAGS_DEF += -march=k8 -m64
	else
		FLAGS_DEF += -DNOT_686
		APPLE := $(shell $(CC) -dM -E - < /dev/null | grep __APPLE__)
		ifneq ($(APPLE),)
		     FLAGS_DEF += -DNOBOOL
		endif
	endif
endif

# FLAGS_DEF are gcc flags that are shared between compiling and
# linking.  CFLAGS_DEF are compile flags, LDFLAGS_DEG are link flags.

# speedy!
#FLAGS_DEF += -O3
#FLAGS_DEF += -fomit-frame-pointer

TEST_OPTION := -fno-math-errno -fno-trapping-math
FLAGS_DEF += $(shell $(CC) $(TEST_OPTION) -c -x c -o $(TMPFILE) - < /dev/null > /dev/null 2> /dev/null && echo "$(TEST_OPTION)")

TEST_OPTION := -fno-stack-protector
FLAGS_DEF += $(shell $(CC) $(TEST_OPTION) -c -x c -o $(TMPFILE) - < /dev/null > /dev/null 2> /dev/null && echo "$(TEST_OPTION)")

