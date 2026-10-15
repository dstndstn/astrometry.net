#!/bin/bash

printhelp() {
  echo ""
  echo "Usage: $0 <options>"
  echo ""
  echo "Required options:"
  echo "  -i <in-dir>: Directory containing all files to be processed."
  echo "  -n <name-scheme>: Shell pattern of input files to match. See documentation for find and its -name option."
  echo "  -o <out-file>: Path to the file with the generated data."
  echo ""
  echo "Optional options:"
  echo "  -l <scale-lower> [default: 0]"
  echo "  -h <scale-higher> [default: 999999]"
  echo "  -c <cpulimit> [default: 300]"
  echo "  -d <downsample> [default: 0]"
  echo "  -b <object-limit>: Corresponds to --objs parameter in solve-field [default: 999999]"
  echo "  -s <source-extractor-config-path>: Path to the configuration file for SourceExtractor."
  echo "    If present, SourceExtractor will be used instead of image2xy."
  echo "  -r <number-of-runs> [default: 1]"
  echo "  -e <command-string>: Allows to call external tools and add their data to the generated file."
  echo "    <command-string> must consist of two parts separated by a colon (:)."
  echo "    The first part is a command to be run, while the second part is the corresponding field name."
  echo "    For each input file found, the given command will be run with the file's path as its argument,"
  echo "      and the command's output will be added to the data generated per file under its fieldname."
  echo "    The time this command takes will not be included in the solving time."
  echo "    Example: ... -e \"./laplacian.sh:laplace-blur\""
  echo "    This option can be present multiple times."
  echo ""
  echo "Example:"
  echo "  $0 -i demo/ -n \"*.jpg\" -o stats.json"
  exit 1
}

if ! eval "solve-field 1>/dev/null"; then
  echo "Trying to execute solve-field returned an error - does the executable exist?"
  exit
fi

intreg="^[0-9]+$"
numreg="^([0-9]+(\.[0-9]+)?|\.[0-9]+)$"

execind=0

while getopts "i:n:o:l:h:c:d:b:s:r:e:" opt; do
  case $opt in
    i)
      if ! [ -d "$OPTARG" ]; then
        echo "ERROR: Input directory \"$OPTARG\" is not a directory."; printhelp
      fi
      indir="$OPTARG"
      ;;
    n)
      namescheme="$OPTARG"
      ;;
    o)
      if [ -e "$OPTARG" ]; then
        echo "ERROR: Output file \"$OPTARG\" already exists."; printhelp
      fi
      outfile="$OPTARG"
      ;;
    l)
      if ! [[ "$OPTARG" =~ $numreg ]] ; then
         echo "ERROR: Lower scale bound (-l) needs to be a number."; printhelp
      fi
      scalelow="$OPTARG"
      ;;
    h)
      if ! [[ "$OPTARG" =~ $numreg ]] ; then
        echo "ERROR: Upper scale bound (-h) needs to be a number."; printhelp
      fi
      scalehigh="$OPTARG"
      ;;
    c)
      if ! [[ "$OPTARG" =~ $intreg ]] ; then
        echo "ERROR: CPU limit (-c) needs to be an integer."; printhelp
      fi
      cpulimit="$OPTARG"
      ;;
    d)
      if ! [[ "$OPTARG" =~ $intreg ]] ; then
        echo "ERROR: Downsampling degree (-d) needs to be an integer."; printhelp
      fi
      downsample="$OPTARG"
      ;;
    b)
      if ! [[ "$OPTARG" =~ $intreg ]] ; then
        echo "ERROR: Object limit (-b) needs to be an integer."; printhelp
      fi
      objs="$OPTARG"
      ;;
    s)
      if ! [ -f "$OPTARG" ]; then
        echo "ERROR: SourceExtractor configuration file \"$OPTARG\" doesn't exist."
        printhelp
      fi
      source_extractor_config="$OPTARG"
      ;;
    r)
      if ! [[ "$OPTARG" =~ $intreg ]] ; then
        echo "ERROR: Amount of repeats (-r) needs to be an integer."; printhelp
      fi
      if [ "$OPTARG" -lt 1 ]; then
        echo "ERROR: Amount of repeats (-r) needs to be at least 1."; printhelp
      fi
      repeats="$OPTARG"
      ;;
    e)
      if [[ "$OPTARG" != *":"* ]]; then
        echo "ERROR: -e requires format \"command:fieldname\" (provided: \"$OPTARG\")."
        printhelp
      fi
      cmds[execind]="${OPTARG%:*}"
      fieldnames[execind]="${OPTARG##*:}"
      execind=$((execind+1))
      ;;
    \?)
      printhelp
      ;;
  esac
done

if [ -z "$indir" ]; then
  echo "Input directory (-i) is required."; printhelp
fi

if [ -z "$namescheme" ]; then
  echo "File name scheme (-n) is required."; printhelp
fi

if [ -z "$outfile" ]; then
  echo "Output file location (-o) is required."; printhelp
fi

if [ -n "$scalelow" ] && [ -n "$scalehigh" ]; then
  if (( $(echo "$scalelow > $scalehigh" | bc -l) )); then
    echo "ERROR: Upper scale bound (-h) needs to be greater than lower scale bound (-l)."
    printhelp
  fi
fi

if [ -z "$scalehigh" ]; then scalehigh=999999; fi
if [ -z "$scalelow" ]; then scalelow=0; fi
if [ -z "$cpulimit" ]; then cpulimit=300; fi
if [ -z "$downsample" ]; then downsample=0; fi
if [ -z "$objs" ]; then objs=999999; fi
if [ -z "$repeats" ]; then repeats=1; fi

if [ "$EUID" -eq 0 ] && [ -z "$AN_ALLOW_ROOT" ]; then
  echo "Script is run at root, but AN_ALLOW_ROOT is not set."
  echo "Please set AN_ALLOW_ROOT to run this script as root."
  exit 1
fi

# Returns value of NDISTRACT field in .match file
ndistract () {
  tmpmatchfn="$1.match.tmp"

  subtable -i "$1.match" -c "NDISTRACT" -o "$tmpmatchfn" 1>/dev/null
  tablist -r "$tmpmatchfn" | xargs
  rm "$tmpmatchfn"
}

# Outputs the number of unrecognized objects
# that are brighter than the dimmest correspondence.
# This is not always equal to NDISTRACT.
bright_unrecognized() {
  tmpcorrfn="$1.corr.tmp"

  subtable -i "$1.corr" -c "field_id" -o "$tmpcorrfn" 1>/dev/null
  corr_field_ids=$(tablist -r "$tmpcorrfn")
  rm "$tmpcorrfn"

  lastid="$(echo "$corr_field_ids" | tail -n 1 | xargs)"
  corr_field_n="$(echo "$corr_field_ids" | wc -l)"
  expected_n=$((lastid+1))
  echo $((expected_n - corr_field_n))
}

mydir="$(dirname "$0")"
cleanscript="$mydir/clean.sh"

cmd="solve-field --overwrite --no-plots -z \"$downsample\" -u app -L \"$scalelow\" -H \"$scalehigh\" --objs \"$objs\""

if [ "$source_extractor_config" ]; then
  cmd="$cmd --use-source-extractor --source-extractor-config $source_extractor_config"
  cmd="$cmd --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending"
fi

# Find out whether a parallelized build is being used
if eval "solve-field --wall-limit 0 1>/dev/null 2>/dev/null"; then
  echo "Notice: solve-field recognized --wall-limit. Using parallel build."
  cmd="$cmd --wall-limit \"$cpulimit\""
else
  if eval "solve-field --cpulimit 0 1>/dev/null 2>/dev/null"; then
    echo "Notice: solve-field did not recognize the option --wall-limit. Assuming serial build."
    cmd="$cmd -l \"$cpulimit\""
  else
    echo "solve-field recognized neither --wall-limit nor --cpulimit."
    echo "Is your executable broken?"
    exit 255
  fi
fi

printf "{\n  \"downsample\": %s,\n  \"scale-low\": %s,\n  \"scale-high\": %s,\n  \"realtime-limit\": %s,\n  \"objs\": %s,\n  \"repeats\": %s"\
  "$downsample" "$scalelow" "$scalehigh" "$cpulimit" "$objs" "$repeats" >> "$outfile"
printf ",\n  \"images\": [" >> "$outfile"

shopt -s lastpipe

inputfiles=$(find "$indir" -mindepth 1 -maxdepth 1 -type f -name "$namescheme" | sort)

# Each array's nth element corresponds to the nth input file.
# Of these arrays, only timesSolvedAr and totalTimeAr have their values modified each run;
# The others, like nsrc, are only written to once.
timesSolvedAr=()
totalTimeAr=()
declare -A cmdResultsAr # [file, cmd]
nsrcAr=()
ncorrAr=()
nbrighterAr=()
ndistractAr=()
scaleAr=()

# Get data
for runind in $(seq "$repeats"); do
  echo "Starting iteration $runind."
  fileind=0
  while read -r file; do
    echo "Solving $file..."

    # Execute all commands from -e
    if [ $runind -eq 1 ]; then
      for i in $(seq 0 $((execind-1))); do
        cmdout="$(${cmds[i]} "$file")"
        cmdResultsAr[$fileind, $i]="$cmdout"
      done
    fi

    t0=$( echo $EPOCHREALTIME | tr -dc "0-9")
    eval "$cmd \"$file\" 2>/dev/null"
    t1=$( echo $EPOCHREALTIME | tr -dc "0-9")
    td=$((t1-t0))

    noext="${file%.*}"
    if [ $runind -eq 1 ]; then
      nsrcAr[fileind]="$(listhead "$noext.axy" | grep "NAXIS2" | xargs | cut -d " " -f3)"
    fi

    if [ -f "$noext.solved" ]; then
      echo "Solved $file in $((td / 1000))ms."
      if [ -z "${timesSolvedAr[fileind]}" ]; then
        timesSolvedAr[fileind]=1
        totalTimeAr[fileind]=$td
        ncorrAr[fileind]="$(listhead "$noext.corr" | grep "NAXIS2" | xargs | cut -d " " -f3)"
        nbrighterAr[fileind]="$(bright_unrecognized "$noext")"
        ndistractAr[fileind]=$(ndistract "$noext")
        scaleAr[fileind]="$(listhead "$noext.wcs" | grep "COMMENT scale: .* arcsec/pix" | cut -d " " -f3)"
      else
        timesSolvedAr[fileind]=$((timesSolvedAr[fileind] + 1))
        totalTimeAr[fileind]=$((totalTimeAr[fileind] + td))
      fi
    else
      echo "Could not solve $file."
    fi

    echo "Cleaning up..."
    $cleanscript "$indir" "$(basename "$noext")" 1>/dev/null

    fileind=$((fileind+1))
  done <<< "$inputfiles"
done

echo "Comparing and writing results to $outfile..."

totalTotalTime=0
totalTimesSolved=0

fileind=0
while read -r file; do

  {
    printf "\n    {"
    printf "\n      \"file\": \"%s\"," "$(basename "$file")"
    printf "\n      \"nsrc\": %s," "${nsrcAr[fileind]}"
  } >> "$outfile"

  # Write command data from -e
  for i in $(seq 0 $((execind-1))); do
    printf "\n      \"%s\": \"%s\"," "${fieldnames[i]}" "${cmdResultsAr[$fileind, $i]}" >> "$outfile"
  done

  if [ -n "${timesSolvedAr[fileind]}" ]; then
    timesSolved="${timesSolvedAr[fileind]}"
    totalTimesSolved=$((totalTimesSolved+timesSolved))
    totalTime="${totalTimeAr[fileind]}"
    totalTotalTime=$((totalTotalTime+totalTime))
    {
      printf "\n      \"solved\": %s," "$timesSolved"
      printf "\n      \"time\": %s," "$((totalTime / timesSolved / 1000))"
      printf "\n      \"ncorr\": %s," "${ncorrAr[fileind]}"
      printf "\n      \"nbrighter\": %s," "${nbrighterAr[fileind]}"
      printf "\n      \"ndistract\": %s," "${ndistractAr[fileind]}"
      printf "\n      \"arcsec/px\": %s" "${scaleAr[fileind]}"
    } >> "$outfile"
  else
    printf "\n      \"solved\": 0" >> "$outfile"
  fi

  printf "\n    }," >> "$outfile"

  fileind=$((fileind+1))
done <<< "$inputfiles"

truncate -s-1 "$outfile" # Remove the last , in the array
printf "\n  ],\n  \"solved\": %s,\n  \"avgtime\": %s,\n  \"notsolved\": %s\n}\n"\
  "$totalTimesSolved" "$((totalTotalTime / totalTimesSolved / 1000))" "$((fileind * repeats - totalTimesSolved))" >> "$outfile"

echo "Done."
