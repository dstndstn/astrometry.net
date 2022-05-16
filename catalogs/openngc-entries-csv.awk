# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

# Skip header line.
NR==1 { next }

{
# skip entries that are neither NGC not IC
if ($1 !~ /^(IC|NGC)[0-9]*$/) next;
# skip "Dup" entries such as NGC4443.
if ($2 ~ /Dup/) next;

# Is it part of NGC or IC?
isngc = ($1 ~ /^NGC/);

# ID number.
id = $1;
gsub(/[A-Z]/, "", id);
id = int(id);

# RA/Dec in degrees.
rahr = substr($3, 1, 2);
ramin = substr($3, 4, 2);
rasec = substr($3, 7, 5);
decsign = substr($4, 1, 1);
decdeg = substr($4, 2, 2);
decmin = substr($4, 5, 2);
decsec = substr($4, 8, 4);

ra  = 15.0 * (rahr + ((ramin + (rasec / 60.0)) / 60.0));
dec = ((decsign == "-") ? -1.0 : 1.0) * (decdeg + ((decmin + (decsec / 60.0)) / 60.0));

# Diameter in arcmins.
size = $6;
if (size == "") {
    size = 0;
}

print isngc ";" id ";" ra ";" dec ";" size;
}
