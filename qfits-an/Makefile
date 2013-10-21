
FILES := anqfits qfits_card qfits_convert qfits_error qfits_header \
	qfits_image qfits_md5 qfits_table qfits_time qfits_tools qfits_byteswap \
	qfits_memory qfits_rw qfits_float md5

OBJS := $(addsuffix .o,$(FILES))
SRCS := $(addsuffix .c,$(FILES))
HEADERS := $(addsuffix .h,$(FILES))

QFITS_LIB_FILE := libqfits.a

all: $(QFITS_LIB_FILE)
.PHONY: all

BASEDIR := ..
COMMON := $(BASEDIR)/util

include $(COMMON)/makefile.common

CFLAGS += $(CFLAGS_DEF)
CFLAGS += -I.

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)

$(QFITS_LIB_FILE): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

install: $(QFITS_LIB_FILE) $(HEADERS)
	@echo Installing in base directory '$(INSTALL_DIR)'
	@for x in $(HEADERS); do \
		echo cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@for x in $(QFITS_LIB_FILE); do \
		echo cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
		cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done
.PHONY: install

clean:
	rm -f $(QFITS_LIB_FILE) $(OBJS) $(DEPS) deps
.PHONY: clean

DEP_OBJ := $(OBJS)

ifneq ($(MAKECMDGOALS),clean)
include $(COMMON)/makefile.deps
endif

