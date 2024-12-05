#!/usr/bin/env bash

# uncensor, decontaminate against test sets, train/test split


export HF_HOME="/projects/bhuang/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="5"

# embedding_model_name_or_path="OrdalieTech/Solon-embeddings-base-0.1"
embedding_model_name_or_path="intfloat/multilingual-e5-base"

# question_file=/projects/bhuang/corpus/text/llm/benchmarks/questions/mt_bench_french.jsonl
question_file=/projects/bhuang/corpus/text/llm/benchmarks/questions/mt_bench.jsonl

input_files=(
    /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b_minscore2.jsonl
    /projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_merged_v01_v02_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated_processed_responded_gpt4turbo0125_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_responded_gpt4turbo0409_processed_chat_merged_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2_respondedgpt4_processed_chat.jsonl
    /projects/bhuang/corpus/text/llm/collected/camel/camel_merged_chemistry_physics_biology_uncensored_deduped_maxsim09_translatedmixtral8x22b_processed_requested_responded_batch_gpt4turbo0409_processed.jsonl
    /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4_merged_v1_processed_chat_responded_gpt4turbo_processed_chat.jsonl
    /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4turbo_processed_chat_requested_responded_batch_gpt4turbo0409_processed_chat.jsonl
    /projects/bhuang/corpus/text/llm/generated/no_robots/no_robots_train_translated_question_translated_system_responded_gpt4_processed_chat.jsonl
    /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_merged_contextqa_direct_reasoning_limited_irrlevant_multihop_responded_titled_processed2_chat.jsonl
    /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated_responded_gpt4turbo_processed_chat.jsonl
    /projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_filteredempty_maxtotlen8192.jsonl
)

input_files=(
    # /projects/bhuang/corpus/text/llm/merged/v3_1/sharegpt4openchat_lmsyschat1m_wildchat_deita_lima_oasst2_airoboros_capybara_codefeedback_codefeedbackfilteredinstruction_agentinstruct_merged_nonemptyturn_totminlen64_instmaxsim095.jsonl
    /projects/bhuang/corpus/text/llm/merged/v3_1/sharegpt4openchat_deita_lima_airoboros_capybara_codefeedback_codefeedbackfilteredinstruction_merged_nonemptyturn_totminlen64_instmaxsim095.jsonl
)

outdir=/projects/bhuang/corpus/text/llm/merged/vigogne-3.2


function process {
    inputfile=$1
    echo $inputfile
    # return

    python scripts/data_processing/uncensor_data.py \
        --input_file $inputfile \
        --output_file ${inputfile%.*}_uncensored.jsonl

    python scripts/data_processing/decontaminate_data.py \
        --question_file $question_file \
        --dataset_file ${inputfile%.*}_uncensored.jsonl \
        --output_dataset_file ${inputfile%.*}_uncensored_decontaminated09.jsonl \
        --embedding_model_name_or_path $embedding_model_name_or_path \
        --batch_size 64 \
        --max_cosine_similarity 0.9

    python scripts/data_processing/split_train_test.py \
        --input_file ${inputfile%.*}_uncensored_decontaminated09.jsonl \
        --output_train_file ${inputfile%.*}_uncensored_decontaminated09_train.jsonl \
        --output_test_file ${inputfile%.*}_uncensored_decontaminated09_test.jsonl \
        --num_test_samples 200

    [ -d $outdir/train ] || mkdir -p $outdir/train
    [ -d $outdir/test ] || mkdir -p $outdir/test

    cp ${inputfile%.*}_uncensored_decontaminated09_train.jsonl $outdir/train
    cp ${inputfile%.*}_uncensored_decontaminated09_test.jsonl $outdir/test

    # clean
    # rm ${inputfile%.*}_uncensored*

}


# process $input_file

for input_file in ${input_files[*]}; do
    process $input_file
done
