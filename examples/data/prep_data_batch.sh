#!/usr/bin/env bash

source .env

# ref
# https://twitter.com/jeffintime/status/1779924149755924707
# https://platform.openai.com/docs/api-reference/batch
# https://community.openai.com/t/batchapi-is-now-available/718416/27?page=2

stage=${1:-0}

input_file=/projects/bhuang/corpus/text/llm/vigogne-alignment-data/magpie-fr-new/magpie_inst_resp_mininstscore28_head50k_1_requested.jsonl
input_file_id="file-WQn5RPwET9Pm419NbkFHeV"
batch_id="batch_674dbb2c49c88190912142a326a9e65b"
output_file_id="file-3maRHFtpodYuZjBQzSoio1"

output_file="${input_file%.*}_responded.jsonl"

if [ $stage -eq 0 ]; then
    echo -e "Upload batch file...\n"
    curl https://api.openai.com/v1/files \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -F purpose="batch" \
        -F file="@$input_file"
fi

if [ $stage -eq 1 ]; then
    echo -e "Create the batch job...\n"
    curl https://api.openai.com/v1/batches \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H 'Content-Type: application/json' \
        --data '{
        "input_file_id": "'"$input_file_id"'",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
        }'
fi

if [ $stage -eq 2 ]; then
    echo -e "Check the status of your job...\n"
    curl https://api.openai.com/v1/batches/$batch_id \
        -H "Authorization: Bearer $OPENAI_API_KEY"
fi

if [ $stage -eq 3 ]; then
    echo -e "Download the completed file!\n"
    curl https://api.openai.com/v1/files/$output_file_id/content \
        -H "Authorization: Bearer $OPENAI_API_KEY" > $output_file
fi

# ---

if [ $stage -eq 7 ]; then
    echo -e "List files...\n"
    curl https://api.openai.com/v1/files \
      -H "Authorization: Bearer $OPENAI_API_KEY"
fi

if [ $stage -eq 8 ]; then
    echo -e "List batches...\n"
    curl https://api.openai.com/v1/batches?limit=2 \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json"
fi

if [ $stage -eq 9 ]; then
    echo -e "Delete files...\n"
    curl https://api.openai.com/v1/files/$file_id \
        -X DELETE \
        -H "Authorization: Bearer $OPENAI_API_KEY"
fi