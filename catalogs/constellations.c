/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <string.h>

#include "constellations.h"

#include "stellarium-constellations.c"

struct shortlong {
    char* shortname;
    char* longname;
};
typedef struct shortlong shortlong_t;

/* 
 lynx -dump -nolist \
 http://www.astro.wisc.edu/~dolan/constellations/abbrevs.html \
 | awk '{L=""; for (i=3;i<=NF;i++) L=L (i>3?" ":"") $i;
 print "{ \"" $2 "\", \"" L "\"},"}'
 */

/*
 Note, these are in alphabetical order, while the Stellarium
 constellations are listed in some other order.
 */
shortlong_t shortlongmap[] = {
    { "And", "Andromeda"},
    { "Ant", "Antlia"},
    { "Aps", "Apus"},
    { "Aqr", "Aquarius"},
    { "Aql", "Aquila"},
    { "Ara", "Ara"},
    { "Ari", "Aries"},
    { "Aur", "Auriga"},
    { "Boo", "Bootes"},
    { "Cae", "Caelum"},
    { "Cam", "Camelopardalis"},
    { "Cnc", "Cancer"},
    { "CVn", "Canes Venatici"},
    { "CMa", "Canis Major"},
    { "CMi", "Canis Minor"},
    { "Cap", "Capricornus"},
    { "Car", "Carina"},
    { "Cas", "Cassiopeia"},
    { "Cen", "Centaurus"},
    { "Cep", "Cepheus"},
    { "Cet", "Cetus"},
    { "Cha", "Chamaeleon"},
    { "Cir", "Circinus"},
    { "Col", "Columba"},
    { "Com", "Coma Berenices"},
    { "CrA", "Corona Austrina"},
    { "CrB", "Corona Borealis"},
    { "Crv", "Corvus"},
    { "Crt", "Crater"},
    { "Cru", "Crux"},
    { "Cyg", "Cygnus"},
    { "Del", "Delphinus"},
    { "Dor", "Dorado"},
    { "Dra", "Draco"},
    { "Equ", "Equuleus"},
    { "Eri", "Eridanus"},
    { "For", "Fornax"},
    { "Gem", "Gemini"},
    { "Gru", "Grus"},
    { "Her", "Hercules"},
    { "Hor", "Horologium"},
    { "Hya", "Hydra"},
    { "Hyi", "Hydrus"},
    { "Ind", "Indus"},
    { "Lac", "Lacerta"},
    { "Leo", "Leo"},
    { "LMi", "Leo Minor"},
    { "Lep", "Lepus"},
    { "Lib", "Libra"},
    { "Lup", "Lupus"},
    { "Lyn", "Lynx"},
    { "Lyr", "Lyra"},
    { "Men", "Mensa"},
    { "Mic", "Microscopium"},
    { "Mon", "Monoceros"},
    { "Mus", "Musca"},
    { "Nor", "Norma"},
    { "Oct", "Octans"},
    { "Oph", "Ophiuchus"},
    { "Ori", "Orion"},
    { "Pav", "Pavo"},
    { "Peg", "Pegasus"},
    { "Per", "Perseus"},
    { "Phe", "Phoenix"},
    { "Pic", "Pictor"},
    { "Psc", "Pisces"},
    { "PsA", "Piscis Austrinus"},
    { "Pup", "Puppis"},
    { "Pyx", "Pyxis"},
    { "Ret", "Reticulum"},
    { "Sge", "Sagitta"},
    { "Sgr", "Sagittarius"},
    { "Sco", "Scorpius"},
    { "Scl", "Sculptor"},
    { "Sct", "Scutum"},
    { "Ser", "Serpens"},
    { "Sex", "Sextans"},
    { "Tau", "Taurus"},
    { "Tel", "Telescopium"},
    { "Tri", "Triangulum"},
    { "TrA", "Triangulum Australe"},
    { "Tuc", "Tucana"},
    { "UMa", "Ursa Major"},
    { "UMi", "Ursa Minor"},
    { "Vel", "Vela"},
    { "Vir", "Virgo"},
    { "Vol", "Volans"},
    { "Vul", "Vulpecula"},
};

int constellations_n() {
    return constellations_N;
}

static void check_const_num(int i) {
    assert(i >= 0);
    assert(i < constellations_N);
}

static void check_star_num(int i) {
    assert(i >= 0);
    assert(i < stars_N);
}

const char* constellations_short_to_longname(const char* shortname) {
    int i;
    int NL = sizeof(shortlongmap) / sizeof(shortlong_t);
    for (i=0; i<NL; i++)
        if (!strcasecmp(shortname, shortlongmap[i].shortname))
            return shortlongmap[i].longname;
    return NULL;
}

const char* constellations_get_longname(int c) {
    check_const_num(c);
    return constellations_short_to_longname(shortnames[c]);
}

const char* constellations_get_shortname(int c) {
    check_const_num(c);
    return shortnames[c];
}

int constellations_get_nlines(int c) {
    check_const_num(c);
    return constellation_nlines[c];
}

il* constellations_get_lines(int c) {
    il* list;
    const int* lines;
    int i;
    check_const_num(c);
    list = il_new(16);
    lines = constellation_lines[c];
    for (i=0; i<2*constellation_nlines[c]; i++) {
        il_append(list, lines[i]);
    }
    return list;
}

il* constellations_get_unique_stars(int c) {
    il* uniq;
    const int* lines;
    int i;
    check_const_num(c);
    uniq = il_new(16);
    lines = constellation_lines[c];
    for (i=0; i<2*constellation_nlines[c]; i++) {
        il_insert_unique_ascending(uniq, lines[i]);
    }
    return uniq;
}

void constellations_get_line(int c, int i, int* ep1, int* ep2) {
    const int* lines;
    check_const_num(c);
    assert(i >= 0);
    assert(i < constellation_nlines[c]);
    lines = constellation_lines[c];
    *ep1 = lines[2*i];
    *ep2 = lines[2*i+1];
}

dl* constellations_get_lines_radec(int c) {
    dl* list;
    const int* lines;
    int i;
    check_const_num(c);
    list = dl_new(16);
    lines = constellation_lines[c];
    for (i=0; i<constellation_nlines[c]*2; i++) {
        int star = lines[i];
        const double* radec = star_positions + star*2;
        dl_append(list, radec[0]);
        dl_append(list, radec[1]);
    }
    return list;
}

void constellations_get_star_radec(int s, double* ra, double* dec) {
    const double* radec;
    check_star_num(s);
    radec = star_positions + s*2;
    *ra = radec[0];
    *dec = radec[1];
}

