
%module plotstuff_c
#undef ATTRIB_FORMAT
#define ATTRIB_FORMAT(x,y,z)
%{
#include "plotstuff.h"
#include "plotoutline.h"
#include "plotgrid.h"
#include "plotindex.h"
#define true 1
#define false 0
%}
%include "plotstuff.h"
%include "plotoutline.h"
%include "plotgrid.h"
%include "plotindex.h"

extern void log_init(int level);

