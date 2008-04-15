{
# Common name: eliminate multiple spaces
namepadded = substr($0, 1, 35);
split(namepadded, parts, " ");
name = "";
for (i=1;; i++) {
	if (!(i in parts)) {
		break;
	}
	if (i > 1) {
		name = name " ";
	}
	name = name parts[i];
}

# NGC or IC?
ic = substr($0, 37, 1);
isic = (ic == "I");
isngc = !isic;

# ID number
idpad = substr($0, 38, 4);
split(idpad, parts, " ");
id = parts[1];

#print "** " name " **" (isic ? "IC" : "NGC") " " id;
if (id > 0) {
	print "{" " .is_ngc = " (isngc ? "TRUE" : "FALSE") ",";
	print "  .id = " id ",";
	print "  .name = \"" name "\"";
	print "},";
}
}
