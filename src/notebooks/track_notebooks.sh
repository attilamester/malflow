#!/bin/bash
find ./ -type f -not -path "*ipynb_checkpoints*" \
  -name "*.ipynb" \
  -not -name "*.GIT-TRACKED.ipynb" -exec sh -c "\
    fname=\$(basename \"{}\" .ipynb); \
    jupyter nbconvert --clear-output \"{}\" --output \"\$fname.GIT-TRACKED.ipynb\"" \;
