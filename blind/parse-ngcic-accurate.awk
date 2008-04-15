# Parse NGC/IC accurate positions from http://www.ngcic.org/corwin/default.htm
{
# NGC or IC?
  ic = substr($0, 1, 1);
  isic = (ic == "I");
  isngc = !isic;

# ID num
  id = substr($0, 2, 4);
  while (substr(id, 1, 1) == "0") {
    id = substr(id, 2);
  }

# extra stuff
  extra = substr($0, 6, 15);
#  if (extra == "               ") {
  match(extra, "^=[?]?[NI][0-9]{4} *$");
  l1 = RLENGTH;
  match(extra, "^[:?]? *$");
  l2 = RLENGTH;
  #print "\"",extra,"\"", l1, l2;
  if ((l1>0) || (l2>0)) {

# RA hr 21:2, min 24:2, sec 27:<=6
    rahrs = substr($0, 21, 2);
    ramins = substr($0, 24, 2);
    i = match(substr($0, 27), "[0-9.]*");
    rasecs = substr($0, 27, RLENGTH);
#print rahrs, ramins, rasecs;
    ra = 15.0 * (rahrs + ramins/60.0 + rasecs/3600.0);
# Dec +- 35:1, deg 36:2, min 39:2, sec 42:?
    decsign = substr($0, 35, 1);
    decdeg = substr($0, 36, 2);
    decmin = substr($0, 39, 2);
    i = match(substr($0, 42), "[0-9.]*");
    decsec = substr($0, 42, RLENGTH);
#print decsign, decdeg, decmin, decsec;
    dec = (decsign == "-" ? -1.0 : 1.0) * (decdeg + decmin/60.0 + decsec/3600.0);
#print ra, dec;
    
    print "{ .is_ngc = " (isngc ? "TRUE" : "FALSE") ",";
    print "  .id = " id ",";
    print "  .ra = " ra ",";
    print "  .dec = " dec ",";
    print "},";
  }
}

