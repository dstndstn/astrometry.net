
/** Copyright 2009 Dustin Lang.
 */

#include <stdint.h>

#include "qfits_std.h"
#include "qfits_config.h"
#include "qfits_image.h"
#include "qfits_byteswap.h"
#include "qfits_error.h"

#define FITSOUTPUTPIXEL(fitstype, ival, optr)	\
	do {										\
		uint8_t* o8;							\
		int16_t* o16;							\
		int32_t* o32;							\
		float*   ofloat;						\
		double*  odouble;						\
		switch (fitstype) {						\
		case BPP_8_UNSIGNED:					\
			o8 = optr;							\
			*o8 = MIN(255, MAX(0, (uint8_t)ival));  \
			break;								\
		case BPP_16_SIGNED:								 \
			o16 = optr;									 \
			*o16 = MIN(INT16_MAX, MAX(INT16_MIN, (int16_t)ival));	\
			break;											\
		case BPP_32_SIGNED:									\
			o32 = optr;										\
			*o32 = MIN(INT32_MAX, MAX(INT32_MIN, (int32_t)ival));	\
			break;											\
		case BPP_IEEE_FLOAT:								\
			ofloat = optr;									\
			*ofloat = ival;									\
			break;											\
		case BPP_IEEE_DOUBLE:								\
			odouble = optr;									\
			*odouble = ival;								\
			break;												\
		default:													\
			qfits_error("Unknown output FITS type %i\n", fitstype); \
			return -1;												\
		}															\
	} while (0)

int qfits_pixel_ctype_size(int ctype) {
	switch (ctype) {
	case PTYPE_DOUBLE:
		return sizeof(double);
	case PTYPE_FLOAT:
		return sizeof(float);
	case PTYPE_INT:
		return sizeof(int);
	case PTYPE_INT16:
		return sizeof(int16_t);
	case PTYPE_UINT8:
		return sizeof(uint8_t);
	}
	return -1;
}

int qfits_pixel_fitstype_size(int fitstype) {
	int n = BYTESPERPIXEL(fitstype);
	if (!n)
		return -1;
	return n;
}

/**
 Converts a value described by a "PTYPE_"
 To a value described by a "BPP_"
 */
int qfits_pixel_ctofits(int ctype, int fitstype,
						const void* cval, void* fitsval) {
	const float* ifloat;
	const double* idouble;
	const int* iint;
	const uint8_t* iu8;
	const int16_t* ii16;

	switch (ctype) {
	case PTYPE_DOUBLE:
		idouble = cval;
		FITSOUTPUTPIXEL(fitstype, *idouble, fitsval);
		break;

	case PTYPE_FLOAT:
		ifloat = cval;
		FITSOUTPUTPIXEL(fitstype, *ifloat, fitsval);
		break;

	case PTYPE_INT:
		iint = cval;
		FITSOUTPUTPIXEL(fitstype, *iint, fitsval);
		break;

	case PTYPE_UINT8:
		iu8 = cval;
		// This generates warnings about "comparison always false due to limited range of data".
		// These warnings are harmless.
		FITSOUTPUTPIXEL(fitstype, *iu8, fitsval);
		break;

	case PTYPE_INT16:
		ii16 = cval;
		// This generates warnings about "comparison always false due to limited range of data".
		// These warnings are harmless.
		FITSOUTPUTPIXEL(fitstype, *ii16, fitsval);
		break;

	default:
		return -1;
	}

	// Byteswap, if necessary.
#ifndef WORDS_BIGENDIAN
	switch (fitstype) {
	case BPP_8_UNSIGNED:
		break;
	case BPP_16_SIGNED:
		qfits_swap_bytes(fitsval, 2);
		break;
	case BPP_32_SIGNED:
	case BPP_IEEE_FLOAT:
		qfits_swap_bytes(fitsval, 4);
		break;
	case BPP_IEEE_DOUBLE:
		qfits_swap_bytes(fitsval, 8);
		break;
	}
#endif
	return 0;
}


#undef FITSOUTPUTPIXEL

