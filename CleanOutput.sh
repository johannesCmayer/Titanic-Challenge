#!/usr/bin/env bash

ARRAY=(./submission_data*.csv, Trained_*_Data)

for filesig in ARRAY; do
  for filename in ./submission_data*.csv; do
    rm ${filename};
    done
  done

exit 0