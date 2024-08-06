awk -F\; -f openngc-entries-csv.awk < NGC.csv > openngc-entries.csv
awk -F\; -f openngc-names-csv.awk < NGC.csv > openngc-names.csv
awk -F\; -f openngc-names-c.awk < openngc-names.csv > openngc-names.c
awk -F\; -f openngc-entries-c.awk < openngc-entries.csv > openngc-entries.c
