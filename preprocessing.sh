#!/usr/bin/env bash
# this script handles the following steps:
# decompression & merging of files in ./releases
# pruning of unnecessary data
# export to flat comment threads as jsonl, corpus.txt & labelstudio compatible .json files

./merge.sh --compress

python3 ./preprocessing/prune.py submissions releases/rsbookclub_submissions.jsonl rsbc_submissions_pruned.jsonl 
python3 ./preprocessing/prune.py comments releases/rsbookclub_comments.jsonl rsbc_comments_pruned.jsonl 

python3 ./preprocessing/flatten.py \
    --submissions rsbc_submissions_pruned.jsonl \
    --comments    rsbc_comments_pruned.jsonl \
    --output      threads.jsonl \
    --export-text         corpus.txt \
    --export-labelstudio  labelstudio_import.json