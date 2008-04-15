VII/118               NGC 2000.0            (Sky Publishing, ed. Sinnott 1988)
================================================================================
NGC 2000.0, The Complete New General Catalogue and Index Catalogue
of Nebulae and Star Clusters by J.L.E. Dreyer
     Sinnott, R.W. (edited by)
    <Sky Publishing Corporation and Cambridge University Press (1988)>
================================================================================
ADC_Keywords: Galaxy catalogs ; Nonstellar objects

================================================================================
Copyright Notice:
    This catalog is copyrighted by Sky Publishing Corporation, which has
    kindly deposited the machine version in the data centers for permanent
    archiving and dissemination to astronomers for scientific research
    purposes only. The data should not be used for commercial purposes
    without the explicit permission of Sky Publishing Corporation.
================================================================================

Description:
    NGC 2000.0 is a modern compilation of the New General Catalogue of
    Nebulae and Clusters of Stars (NGC), the Index Catalogue (IC), and the
    Second Index Catalogue compiled by J. L. E. Dreyer (1888, 1895, 1908).
    The new compilation of these classical catalogs is intended to meet
    the needs of present-day observers by reporting positions at equinox
    B2000.0 and by incorporating the corrections reported by Dreyer
    himself and by a host of other astronomers who have worked with the
    data and compiled lists of errata. The object types given are those
    known to modern astronomy. The catalog lists object ID, object type,
    positions in equinox B2000.0, source of modern data (see NGC 2000
    paperback copy), constellation, object size, magnitude, and the
    description of the object as given by Dreyer. The order of the new
    catalog is strictly by right ascension, the NGC and IC objects being
    merged into one machine-readable file.

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl    Records    Explanations
--------------------------------------------------------------------------------
ReadMe          80          .    This file
ngc2000.dat     96      13226    The NGC 2000.0 Catalogue
names.dat       70        227    Index of Messier and common names
--------------------------------------------------------------------------------

Byte-per-byte Description of file: ngc2000.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  5  A5     ---     Name     NGC or IC designation (preceded by I)
   7-  9  A3     ---     Type    *Object classification
  11- 12  I2     h       RAh      Right Ascension 2000 (hours)
  14- 17  F4.1   min     RAm      Right Ascension 2000 (minutes)
      20  A1     ---     DE-      Declination 2000 (sign)
  21- 22  I2     deg     DEd      Declination 2000 (degrees)
  24- 25  I2     arcmin  DEm      Declination 2000 (minutes)
      27  A1     ---     Source  *Source of entry
  30- 32  A3     ---     Const    Constellation
      33  A1     ---     l_size   [<] Limit on Size
  34- 38  F5.1   arcmin  size     ? Largest dimension
  41- 44  F4.1   mag     mag      ? Integrated magnitude, visual or photographic
                                      (see n_mag)
      45  A1     ---     n_mag    [p] 'p' if mag is photographic (blue)
  47- 96  A50    ---     Desc    *Description of the object
--------------------------------------------------------------------------------
Note on Type: the field is coded as follows:
     Gx    Galaxy
     OC    Open star cluster
     Gb    Globular star cluster, usually in the Milky Way Galaxy
     Nb    Bright emission or reflection nebula
     Pl    Planetary nebula
     C+N   Cluster associated with nebulosity
     Ast   Asterism or group of a few stars
     Kt    Knot  or  nebulous  region  in  an  external galaxy
     ***   Triple star
     D*    Double star
     *     Single star
     ?     Uncertain type or may not exist
     blank Unidentified at the place given, or type unknown
     -     Object called nonexistent in the RNGC (Sulentic and Tifft 1973)
     PD    Photographic plate defect

Note on Source: sources that have been used to correct or update
    modern data in NGC 2000.0 (type, positions, magnitude, and size).
    Uppercase letters denote special NGC and IC errata lists, which have
    usually been accorded more weight than the source catalogues
    themselves. In parentheses after each citation is the number of times
    it has been used to update NGC entries (first number) and those in
    the IC (second number).
    A   Archinal, Brent A. Version 4.0 of an unpublished list of errata to
        the RNGC, dated March 19, 1987. (110,0)
    a   Arp, H., "Atlas of Peculiar Galaxies", 1966ApJS...14....1A (1,2)
        (Catalog <VII/74>)
    c   Corwin, Harold G., Jr., A. de Vaucouleurs, and G. de Vaucouleurs,
        "Southern Galaxy Catalogue", Austin, Texas: University of Texas
        Monographs in Astronomy No. 4, 1985. (152,564)
        (Catalog <VII/116>)
    d   Dreyer, J.L.E., New General Catalogue of Nebulae and Clusters of
        Stars (1888), Index Catalogue (1895), Second Index Catalogue (1908).
        London: Royal Astronomical Society, 1953. (28,2157)
    D   Dreyer, J.L.E., ibid. Errata on pages 237, 281-283, and 366-378.
        (158,28)
    F   Skiff, Brian, private communication of February 27, 1988.  (93,36)
    h   Holmberg, E., "A Study of Double and Multiple Galaxies",
        Lund Annals, 6, 1937. (13,2)
    k   Karachentsev, I.D., "A Catalogue of Isolated Pairs of Galaxies
        in the Northern Hemisphere"; also, Karachentseva, V.E.,
        "A Catalog of Isolated Galaxies." Astrofiz. Issled. Izv. Spetz.
        Astrofiz., 7, 3, 1972, and 8, 3, 1973. (0,4)
        (Catalogs <VII/77>, <VII/82>, <VII/83>)
    m   Vorontsov-Velyaminov, B.A., and V.P. Arhipova,
        "Morphological Catalog of Galaxies", Parts I-V.
        Moscow: Moscow State University, 1962-74. (9,679)
        (Catalogs <VII/62> and <VII/100>)
    n   Reinmuth, K., "Photographische Positionsbestimmung von NebelRecken"
        Veroff der Sternwarte zu Heidelberg, several papers, 1916-40. (0,4)
    o   Alter, G., B. Balazs, and J. Ruprecht, Catalogue of Star Clusters
        and Associations, 2nd edition.  Budapest: Akademiai Kiado, 1970. (5,0)
        (Catalogs <VII/5>, <VII/44> and <VII/101>)
    r   Sulentic, Jack W., and William G. Tifft, "The Revised New General
        Catalogue of Nonstellar Astronomical Objects (RNGC)".
        Tucson, Arizona:University of Arizona Press, 1973. (4016,0)
        (Catalog <VII/1>)
    s   Hirshfeld, Alan, and Roger W. Sinnott, eds., Sky Catalogue 2000.0,
        Vol.2, Cambridge, Massachusetts:
        Sky Publishing Corp. and Cambridge University Press, 1985. (3098,238)
    t   Tully, R.B., "Nearby Galaxies Catalog". New York: Cambridge
        University Press, 1988.
        A preliminary version on magnetic tape (1981) was used here. (23,17)
        (Catalog <VII/145>)
    u   Nilson P.N., Uppsala Ceneral Catalogue of Galaxies.
        Uppsala: Uppsala Astronomical Observatory, 1973. (15,543)
        (Catalog <VII/26>)
    v   de Vaucouleurs, G., A. de Vaucouleurs, and H.C. Corvin, Jr.,
        Second Reference Catalogue of Bright Galaxies. Austin, Texas,
        University of Texas Press, 1976.(118,206)
        (Catalog <VII/112>)
    x   Dixon, R.S., and George Sonneborn, "A Master List of Nonstellar
        Optical Astronomical Objects (MOL)".  Columbus, Ohio,
        Ohio State University Press, 1980.
        It should be noted that most of the information for codes
        a,h,k,m,n,o,u and z was extracted from the magnetic-tape
        version of this catalogue.
        The x code refers to IC objects identified in a literature
        search by these authors. (0,526)
    z   Zwicky, F., E. Herzog, and P. Wild, "Catalogue of Galaxies and
        Clusters of Galaxies", Vol.I. Pasadena, Calif., California Institute
        of Technology, 1961. Also, successive volumes through 1968. (1,380)
        (Catalog <VII/49>)

Note on Desc: description of the object, as given by Dreyer or
     corrected by him, in a coded or abbreviated form. The abbreviations
     and their combination are fully described in the introduction
     to the published catalog.
     ab       about
     alm      almost
     am       among
     annul    annular or ring nebula
     att      attached
     b        brighter
     bet      between
     biN      binuclear
     bn       brightest to n side
     bs       brightest to s side
     bp       brightest to p side
     bf       brightest to f side
     B        bright
     c        considerably
     chev     chevelure
     co       coarse, coarsely
     com      cometic (cometary form)
     comp     companion
     conn     connected
     cont     in contact
     C        compressed
     Cl       cluster
     d        diameter
     def      defined
     dif      diffused
     diffic   difficult
     dist     distance, or distant
     D        double
     e        extremely, excessively
     ee       most extremely
     er       easily resolvable
     exc      excentric
     E        extended
     f        following (eastward)
     F        faint
     g        gradually
     glob.    globular
     gr       group
     i        irregular
     iF       irregular figure
     inv      involved, involving
     l        little (adv.); long (adj.)
     L        large
     m        much
     m        magnitude
     M        middle, or in the middle
     n        north
     neb      nebula
     nebs     nebulous
     neby     nebulosity
     nf       north following
     np       north preceding
     ns       north-south
     nr       near
     N        nucleus, or to a nucleus
     p        preceding (westward)
     pf       preceding-following
     p        pretty (adv., before F. B. L, S)
     pg       pretty gradually
     pm       pretty much
     ps       pretty suddenly
     plan     planetary nebula (same as PN)
     prob     probably
     P        poor (sparse) in stars
     PN       planetary nebula
     r        resolvable (mottled, not resolved)
     rr       partially resolved, some stars seen
     rrr      well resolved, clearly consisting of stars
     R        round
     RR       exactly round
     Ri       rich in stars
     s        suddenly (abruptly)
     s        south
     sf       south following
     sp       south preceding
     sc       scattered
     sev      several
     st       stars (pl.)
     st 9...  stars of 9th magnitude and fainter
     st 9..13 stars of mag. 9 to 13
     stell    stellar, pointlike
     susp     suspected
     S        small in angular size
     S*       small (faint) star
     trap     trapezium
     triangle triangle, forms a triangle with
     triN     trinuclear
     v        very
     vv       _very_
     var      variable
     *        a single star
     *10      a star of 10th magnitude
     *7-8     star of mag. 7 or 8
     **       double star (same as D*)
     ***      triple star
     !        remarkable
     !!       very much so
     !!!      a magnificent or otherwise interesting object
--------------------------------------------------------------------------------

Byte-by-byte Description of file: names.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 35  A35   ---     Object    Common name (including Messier numbers)
  37- 41  A5    ---     Name     *NGC or IC name, as in ngc2000.dat
  43- 70  A28   ---     Comment   Text of comment, if any
--------------------------------------------------------------------------------
Note on Name: this field may be blank for Messier objects without
     NGC or IC counterparts.
     when one object corresponds to several entries in ngc2000,
     the Object is repeated (e.g. Copeland's Septet appears 7 times)
--------------------------------------------------------------------------------

History by Wayne H. Warren Jr., December 1989:
        It is important, even for users of the machine-readable catalog
    and this documentation, to also have a copy of the published book.
    In addition to the tables and reference sources mentioned in this
    document, the book provides an introductory section with a brief
    history of the NGC and IC catalogs, a count of objects by
    constellation, information on Dreyer's descriptions, a table cross
    index of Messier and NGC/IC designations, and a table of common names
    for NGC objects. The book also contains a table of right ascensions
    for NGC and IC objects.
        A magnetic tape containing NGC 2000.0 was received from
    William E. Shawcross of Sky Publishing Corporation on August 14, 1989.
    According to Mr. Shawcross, the file supplied to the ADC was an
    unmodified version of the one used to produce the book, and it still
    contained the TEX commands employed to produce the special symbols
    present in the printed version. As received, the file also contained a
    single copyright text record at its beginning. The text record was
    removed to an added first file in the archived version and
    supplemented with a small amount of additional information. The TEX in
    the data file was replaced by standard characters to represent the
    information. Special symbols, such as "\Delta", "\bigcirc", etc., were
    changed to their spelled-out equivalents.
        The size field was modified to add decimal points to integer
    numbers and to align all values properly so that the field can be
    processed with a single format specification. The magnitude field was
    modified by moving the "p" code for photographic magnitude to its own
    byte in order to remove it from the numerical field. Decimal points
    were added to all integer numbers in this field also.
        The catalog data file was run through the ADC General
    Verification Program, which checks data ranges and for various other
    problems that can be detected in a systematic way.

Further history:
    The standardised document was generated in April 1977 at CDS
    (James Marcout, Francois Ochsenbein).

Acknowledgements:
    Appreciation is expressed to William E. Shawcross for responding to a
    request from the ADC to make NGC 2000.0 available to the scientific
    community in machine-readable form. Mr. Shawcross also arranged for a
    copy of the machine-readable TEX file to be created for deposit in the
    archives of the data centers. I am grateful to both Mr. Shawcross and
    to Roger W. Sinnott for reviewing a draft copy of this document and
    making comments. The comments resulted in the finding and elimination
    of a few TEX symbols that were missed during the initial work.

    The meticulous documentation initiated by Wayne H. Warren at ADC
    (December 1989) is the basis of the present document.

References:
   Dreyer, J.L.E. 1888, "New General Catalogue of Nebulae and Clusters of
      Stars", MmRAS, 49, Part I (reprinted 1953, London: Royal Astronomical
      Society)
   Dreyer, J.L.E. 1895, "Index Catalogue of Nebulae Found in the Years 1888 to
      1894 with Notes and Corrections to the New General Catalogue", MmRAS, 51,
      185 (London: Royal Astronomical Society, reprinted 1953)
   Dreyer, J.L.E. 1908, "Second Index Catalogue of Nebulae Found in the Years
      1895 to 1907; with Notes and Corrections to the New General Catalogue
      and to the Index Catalogue for 1888 to 1894", MmRAS, 59, Part 2, 105
      (London: Royal Astronomical Society, reprinted 1953)
   Sulentic, J. W. and Tifft, W. G. 1973, "The Revised New General
      Catalogue of Nonstellar Astronomical Objects"
     (Tucson: The University of Arizona Press; catalog <VII/1>).
================================================================================
(End)                                     Francois Ochsenbein [CDS]  03-Apr-1997
