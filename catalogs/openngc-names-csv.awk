# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

function printName(name) {
    print isngc ";" id ";" name;
}

# Skip header line.
NR==1 { next }

{
if ($1 !~ /^(IC|NGC)[0-9]*$/) next;

# Is it part of NGC or IC?
isngc = ($1 ~ /^NGC/);

# ID number.
id = $1;
gsub(/[A-Z]/, "", id);
id = int(id);

# Common names (comma separated).
split($24, names, ",");
for (i in names) {
    printName(names[i]);
}

# Add Messier number as a common name.
if ($19 != "") {
    printName("M " int($19));
}
}
