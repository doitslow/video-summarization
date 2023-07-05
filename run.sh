#!/bin/bash
CODE=/home/john/Desktop/P16/0-code/text2clip/
DATA=/home/john/Desktop/P16/1-data/MediaCorp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd $CODE
#DIN=/home/john/Desktop/P16/1-data/MediaCorp/mediacorp-test-set
#DIN=/home/john/Desktop/P16/1-data/MediaCorp/mediacorp-test-set/The_Wish
DIN=/home/john/Desktop/P16/1-data/MediaCorp/mediacorp-test-set/Local_Fine_Produce

ARGS=(
"--din $DIN"
"--method keys_ensemble"
#"--method keysen_ensemble"
"--match_batch 64" # consider increase to a much larger value
"--caption_batch 64" # consider increase to a much larger value
"--clean_batch 8" # consider increase to a much larger value
"--cleaning c2"
"--aes_path $DATA/aes "
"--meta_path $DATA/Metadata.xlsx"
"--post_hecate"
)

python select_shot.py ${ARGS[@]}