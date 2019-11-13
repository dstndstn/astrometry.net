# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

{
print "{ .is_ngc = " ($1 ? "TRUE" : "FALSE") ",";
print "  .id = " $2 ",";
print "  .ra = " $3 ",";
print "  .dec = " $4 ",";
print "  .size = " $5 ",";
print "},";
}
