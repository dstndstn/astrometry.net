{
# Is it part of NGC or IC?
ic = substr($0, 1, 1);
isic = (ic == "I");
isngc = !isic;

# ID number
id = substr($0, 2, 4);

# Classification
classpad = substr($0, 7, 3);
split(classpad, parts, " ");
class = parts[1];

# Location
rahrs = substr($0, 11, 2);
ramins = substr($0, 14, 4);
decsign = substr($0, 20, 1);
decdeg = substr($0, 21, 2);
decmin = substr($0, 24, 2);

ra  = 15.0 * (rahrs + ramins / 60.0);
dec = ((decsign == "-") ? -1.0 : 1.0) * (decdeg + (decmin / 60.0));

# Size in arcmin
sizepad = substr($0, 34, 5);
split(sizepad, parts, " ");
size = parts[1];
if (size == "") {
	size = 0;
}

constellation = substr($0, 30, 3);

print "{ 'is_ngc': " (isngc ? "True" : "False") ",";
print "  'id': " id ",";
print "  'classification': \"" class "\",";
print "  'ra': " ra ",";
print "  'dec': " dec ",";
print "  'size': " size ",";
print "  'constellation': \"" constellation "\",";
print "},";
}
