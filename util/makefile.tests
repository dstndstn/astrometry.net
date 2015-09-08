# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE


### First, define these variables in your Makefile.

ALL_TEST_FILES ?=
ALL_TEST_EXTRA_OBJS ?=
ALL_TEST_LIBS ?=
ALL_TEST_EXTRA_LDFLAGS ?= -lm

### Next, include this file.

### Ignore everything below here.  It's just magic.

ALL_TEST_FILES_C = $(addsuffix .c, $(ALL_TEST_FILES))
ALL_TEST_FILES_O = $(addsuffix .o, $(ALL_TEST_FILES))
ALL_TEST_FILES_MAIN_C = $(addsuffix -main.c, $(ALL_TEST_FILES))
ALL_TEST_FILES_MAIN_O = $(addsuffix -main.o, $(ALL_TEST_FILES))

ALL_TESTS_CLEAN = $(ALL_TEST_FILES) $(ALL_TEST_FILES_O) \
    $(ALL_TEST_FILES_MAIN_C) $(ALL_TEST_FILES_MAIN_O) \
    test test.o test.c cutest.o

MAKE_TESTS := $(COMMON)/make-tests.sh

$(COMMON)/make-tests.sh: ;

$(ALL_TEST_FILES_MAIN_C): %-main.c: $(MAKE_TESTS)
$(ALL_TEST_FILES_MAIN_C): %-main.c: %.c
	$(AN_SHELL) $(MAKE_TESTS) $^ > $@

$(ALL_TEST_FILES): %: %-main.o %.o $(COMMON)/cutest.o

TEST_SOURCES = $(ALL_TEST_FILES_C)
test.c: $(MAKE_TESTS) $(TEST_SOURCES)
	$(AN_SHELL) $(MAKE_TESTS) $(TEST_SOURCES) > $@

test: test.o $(COMMON)/cutest.o $(ALL_TEST_FILES_O) $(sort $(ALL_TEST_EXTRA_OBJS)) $(ALL_TEST_LIBS)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(ALL_TEST_EXTRA_LDFLAGS) $(LDLIBS)

$(ALL_TEST_FILES_O) $(ALL_TEST_FILES_MAIN_O) test.o: %.o: %.c
	$(CC) -o $@ $(CPPFLAGS) $(CFLAGS) -c $< -I$(COMMON)
