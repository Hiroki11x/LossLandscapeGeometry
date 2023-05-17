#!/bin/sh
files=$(find . -name "n*tar" | sed 's/\.\/\(.*\)\.tar/\1/' | awk '{ print $1 }' | tr '\n' ' ' | head -n 1 | awk '1')
for filepath in ${files}
do
  filename=`basename ${filepath} .tar`
  if [ ! -e ${filename}.tar ]; then
    continue
  fi
  mkdir ${filename}
  tar -xf ${filename}.tar -C ${filename}
  rm ${filename}.tar
done