# Tools

This directory contains additional tools that are not necessary to run
or build Astrometry.net, but might still be useful.

## `add-distractors.sh`
Adds distractors to an image.
Size and amount of these distractors are configurable.
Optionally, an image file to be used as a distractor can be supplied.
A file specifying the positions of the distractors can also be supplied.

This can be useful to automatically generate test data to test the resilience
of Astrometry.net against images with lots of distractors.

Usage: `add-distractors.sh
    ( <DOT-SIZE> | <DOT-SRC-FILE> )
    ( <NUM-DISTRACTORS> | <DISTRACTORS-POS-FILE> )
    <SRC-IMG> <DEST-IMG>`

When the first argument is an integer, it is used as the size,
in pixels, of the distractors. In this case, the distractors are white dots -
more specifically, radial gradients, which means they are fully opaque at their
centre while fading into full transparency on the outside.
When the first argument is not an integer, it is used as the path
to the image that will be overlaid onto the original image
instead of the white dots.

When the second argument is an integer, it is used as the amount of distractors
to add to the image.
In this case, the positions of the distractors are generated randomly.
When the second argument is not an integer, it is used as the path
to the file specifying the positions of the distractors.
In this case, each line in the file will cause a distractor to be placed.
Lines must have the format `([+][0-9]+){2}`.
For example, the line `+500+200` would place a distractor at X 500 Y 200.

### Prerequisites:
- [ImageMagick](https://imagemagick.org/) for the `convert` command
- `/dev/urandom`

### Examples:
- `tools/add-distractors.sh 12 32
demo/apod5.jpg demo/apod5-with-distractors.png`
- `tools/add-distractors.sh 12 tools/add-distractors-demo-positions.txt
demo/apod5.jpg demo/apod5-with-distractors.png`

The script is POSIX compatible.

## `benchmark.sh`
Executes `solve-field` on a bunch of files that match a specified pattern
in a specified directory, gathers a bunch of statistics about them,
and writes those to a specified output file in JSON.

Usage: `benchmark.sh <OPTIONS>`

Example: `benchmark.sh -i ../demo/ -n "*.jpg" -o ../demo/statistics.json`

Required options:
- `-i`: Input directory where all files to be solved are located.
- `-n`: Shell pattern of input files to match.
`solve-field` will be executed on all files that are 
located within the input directory and match this pattern.
See documentation for `find` and its `-name` option.
- `-o`: Path to the output file that will be generated.

Optional options:
- `-l`: Lower scale bound, in arcseconds per pixel (`--scale-low`).
Default 0.
- `-h`: Upper scale bount, in arcseconds per pixel (`--scale-high`).
Default 999999.
- `-c`: CPU time limit (`--cpulimit`). Default 300.
- `-d`: Downsampling value (`--downsample`). Default 0.
- `-b`: Object limit (`--objs`). Default 999999.
- `-s`: Path to a configuration file for SourceExtractor.
    If present, SourceExtractor will be used instead of image2xy.
- `-r`: Number of runs. Measured time is averaged between all successful runs.
    Since this essentially runs the benchmark multiple times, it can be useful
    to mitigate random, temporary deviations in performance.
    Default 1.
- `-e`: Allows to execute external tools
    and add their output to the generated file.
    This option's argument must consist of two parts separated by a colon (:).
    The first part is a command to be run,
    while the second part is the corresponding field name.
    For each input file found, the given command will be run
    with the file's path as its argument, and the command's output will be
    added to the data generated per file under its field name.
    For example, `-e \"./laplacian.sh:laplace-blur\"` would execute a script
    located at `./laplacian.sh` on every input file before `solve-field` is
    called and save the output of that file to a field named `laplace-blur`
    in the generated output file. This option can be present multiple times.
    The time this command takes will not be included in the total solving time.

Per file, it gathers the following statistics:
- `solved`: How many times the file has been solved.
    This will be a number between 0 and the number of runs (see option `-r`).
- `time` (if solved): The time it took for the file to be solved
- `nsrc`: The number of sources extracted from the file
- `ncorr` (if solved): The number of correspondences, 
    i.e. sources that could be mapped to objects present in the index,
    i.e. stars recognized in the image
- `nbrighter` (if solved): The number of sources that are not correspondences,
    but are brighter than the dimmest correspondence found
- `ndistract` (if solved): The value of the NDISTRACT field
    in the corresponding .match file.
    This is not necessarily equal to `nbrighter`.

The output file also contains the `downsample`, `scale-low`, `scale-high`,
`cpulimit`, `objs` and `repeats` fields
(i.e. the configuration options of the benchmark),
the amount of times an input file could and could not be solved
(`solved` and `notsolved`), as well as
the average time the solved images took (`avgtime`).

### Prerequisites:
- `solve-field` in any of the directories in your `$PATH`
- Astrometry.net's FITS utilities `tablist`, `listhead` and `subtable`
    in any of the directories in your `$PATH`
- `clean.sh` in the same directory as this script (see below)

This script is not POSIX compatible
because it uses `shopt -s lastpipe` and `$EPOCHREALTIME`.

## `clean.sh`

Removes all files automatically generated by `solve-field`,
such as `.axy` or `.solved` files.

Usage: `clean.sh <PATH> <FILE-STEM>`

This script removes all files that begin with the given file stem
and match the naming scheme of files automatically generated by Astrometry.net,
so **be careful!**

POSIX compatible.
