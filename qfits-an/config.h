/* Version number of package */
#define QFITS_VERSION "6.2.0"

#include "an-endian.h"

#if IS_BIG_ENDIAN
#define WORDS_BIGENDIAN 1
#else
#undef WORDS_BIGENDIAN
#endif

