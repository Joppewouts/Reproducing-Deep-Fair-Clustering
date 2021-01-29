#!/bin/bash

die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in pdoc3; do
    command -v "$cmd" >/dev/null ||
        die "Missing $cmd; \`pip install $cmd\`"
done

echo "top"

DOCROOT=$(pwd)
BUILDROOT="$DOCROOT/doc/build"


echo
echo 'Building documentation'
echo
mkdir -p "$BUILDROOT"

echo $DOCROOT
echo $BUILDROOT
echo $(dirname "$(pwd)")

pdoc --html --output-dir "$BUILDROOT" --skip-errors "$DOCROOT"