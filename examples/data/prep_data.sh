#!/usr/bin/env bash

source .env

# instruct data log
# python vigogne/data/merge_datasets.py --inputs-files /projects/bhuang/corpus/text/llm/alpaca/alpaca_data_cleaned_fr_52k_train.jsonl /projects/bhuang/corpus/text/llm/self_instruct/self_instruct_data_final_100k_new.jsonl data/instruct/dolly_bactrian_fr_15k.jsonl --output-file data/instruct/tmp_alpaca_selfinstruct_dollybactrian.jsonl
# python vigogne/data/merge_datasets.py --inputs-files data/instruct/tmp_alpaca_selfinstruct_dollybactrian.jsonl /projects/bhuang/corpus/text/llm/grade_school_math_instructions/grade_school_math_instructions.jsonl /projects/bhuang/corpus/text/llm/logic_inference_oa/logic_inference_oa_10k.jsonl /projects/bhuang/corpus/text/llm/hc3/hc3_dedup_10k.jsonl /projects/bhuang/corpus/text/llm/code_alpaca/code_alpaca_20k.jsonl /projects/bhuang/corpus/text/llm/evol_instruct/evol_instruct_143k_max512_50k.jsonl --output-file data/instruct/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl

# chat data log
# python vigogne/data/convert_alpaca_to_chat.py data/instruct/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl
# python vigogne/data/convert_alpaca_to_chat.py data/instruct/tmp_alpaca_data_cleaned_fr_52k_test.jsonl data/chat/tmp_alpaca_data_cleaned_fr_52k_test.jsonl
# python vigogne/data/merge_datasets.py --inputs-files data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl /projects/bhuang/corpus/text/llm/self_chat/self_chat_data_quora_fr_50k.jsonl /projects/bhuang/corpus/text/llm/oasst/oasst_20230412_fr_top1.jsonl /projects/bhuang/corpus/text/llm/oasst/oasst_20230412_en_top1.jsonl /projects/bhuang/corpus/text/llm/sharegpt_90k/sg_90k_all_cleaned_fr.jsonl /projects/bhuang/corpus/text/llm/sharegpt_90k/sg_90k_all_cleaned_en.jsonl /home/bhuang/nlp/vigogne/data/chat/dummy_chat.jsonl /projects/bhuang/corpus/text/llm/baize/alpaca_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/medical_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/quora_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/stackoverflow_chat_data_20k.jsonl --output-file data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct_selfchatquora_oasstfren_sgfren_dummy_baize.jsonl

# python vigogne/data/random_split.py /projects/bhuang/corpus/text/llm/baize/alpaca_chat_data.jsonl /projects/bhuang/corpus/text/llm/baize/alpaca_chat_data_10k_rem.jsonl /projects/bhuang/corpus/text/llm/baize/alpaca_chat_data_10k.jsonl 10000
# python vigogne/data/random_split.py /projects/bhuang/corpus/text/llm/baize/medical_chat_data.jsonl /projects/bhuang/corpus/text/llm/baize/medical_chat_data_10k_rem.jsonl /projects/bhuang/corpus/text/llm/baize/medical_chat_data_10k.jsonl 10000
# python vigogne/data/random_split.py /projects/bhuang/corpus/text/llm/baize/quora_chat_data.jsonl /projects/bhuang/corpus/text/llm/baize/quora_chat_data_10k_rem.jsonl /projects/bhuang/corpus/text/llm/baize/quora_chat_data_10k.jsonl 10000
# python vigogne/data/random_split.py /projects/bhuang/corpus/text/llm/baize/stackoverflow_chat_data.jsonl /projects/bhuang/corpus/text/llm/baize/stackoverflow_chat_data_20k_rem.jsonl /projects/bhuang/corpus/text/llm/baize/stackoverflow_chat_data_20k.jsonl 20000

# translate open orca's instructions
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "facebook/nllb-200-3.3B" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_nllb200_3b3_fp16.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "facebook/nllb-200-1.3B" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_nllb200_1b3.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "facebook/m2m100_1.2B" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_m2m100_1b2.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "facebook/m2m100_1.2B" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_m2m100_1b2_fp16.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "facebook/m2m100_418M" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_m2m100_418m.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --max_samples 100000 --model_name_or_path "facebook/m2m100_418M" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_100k_fr_m2m418m.jsonl
# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --model_name_or_path "Helsinki-NLP/opus-mt-en-fr" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_helsinki.jsonl

# python vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/3_5M-GPT3_5-Augmented.parquet --field_names '["question"]' --max_samples 200000 --model_name_or_path "facebook/m2m100_418M" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_100k_fr_m2m418m.jsonl
# python /vigogne/data/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/openorca_gpt4_50k_1.parquet --field_names '["question"]' --model_name_or_path "facebook/m2m100_418M" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/openorca_gpt4_50k_fr_m2m418m_1.jsonl

# export GOOGLE_PROJECT_ID="focal-woods-393916"
# python vigogne/data/translate_dataset_google_api.py --dataset_name Open-Orca/OpenOrca --dataset_split_name train --field_names '["question"]' --max_samples 200000 --num_workers 64 --output_parquet /projects/bhuang/corpus/text/llm/open_orca/open_orca_fr_google_100k.parquet --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_fr_google_100k.jsonl
# python vigogne/data/translate_dataset_google_api.py --dataset_name /projects/bhuang/corpus/text/llm/open_orca/1M-GPT4-Augmented.parquet --field_names '["question"]' --max_samples 50000 --num_workers 1 --output_file /projects/bhuang/corpus/text/llm/open_orca/open_orca_gpt4_fr_google.jsonl

# orca
# gpt4
# 38.89 + 119.23 + 78.10 = 236.22 (gpt4 10k)
#  python scripts/data_generation/generate_instruction_following_samples.py --input_json_file /projects/bhuang/corpus/text/llm/open_orca/openorca_gpt4_100k_fr_m2m1b2_normalized.jsonl --output_json_file /projects/bhuang/corpus/text/llm/open_orca/openorca_gpt4_100k_fr_m2m1b2_normalized_responded.jsonl --system_field system_prompt --instruction_field translated_question --response_field fr_response --model gpt-4-0314 --max_parallel_requests 1 --max_samples 10000
# 8.74 + 155.95 + 105.02 = 269.71 (gpt4 10k (shorter) + gpt3.5 71791)

# python /home/bhuang/nlp/vigogne/scripts/data_processing/translate_dataset.py --dataset_parquet_file /projects/bhuang/corpus/text/llm/open_orca/1m_gpt4_augmented_deduped_lot2_max1024_50k_part3.parquet --field_names '["question"]' --model_name_or_path "facebook/m2m100_1.2B" --fp16 "True" --batch_size "8" --translate_after_split "True" --output_file /projects/bhuang/corpus/text/llm/open_orca/1m_gpt4_augmented_deduped_lot2_max1024_50k_part3_fr_m2m1b2.jsonl

# $6 for translation (multi times)
# python scripts/data_processing/translate_dataset_chatgpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/open_orca/lot2/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/collected/open_orca/lot2/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35.jsonl \
#     --column_name question \
#     --max_samples 5000 \
#     --max_parallel_requests 16

# $88 for gen
# python scripts/data_generation/generate_responses.py \
#     --input_json_file /projects/bhuang/corpus/text/llm/collected/open_orca/lot2/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35_normalized.jsonl \
#     --output_json_file /projects/bhuang/corpus/text/llm/collected/open_orca/lot2/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35_normalized_respondedgpt4.jsonl \
#     --system_field translated_system_prompt \
#     --instruction_field translated_question \
#     --response_field response_on_translated_prompt \
#     --model gpt-4-0314 \
#     --max_parallel_requests 4

# python scripts/data_processing/translate_dataset_chatgpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35.jsonl \
#     --column_name query \
#     --max_samples 5000 \
#     --max_parallel_requests 16

# $101 for gen
# python scripts/data_generation/generate_responses.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35_processed_prompted.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35_processed_prompted_respondedgpt4.jsonl \
#     --system_field system_prompt \
#     --instruction_field translated_query \
#     --response_field response_on_translated_prompt \
#     --model gpt-4-0314 \
#     --temperature 0 \
#     --max_parallel_requests 4

# python scripts/data_processing/translate_dataset_chatgpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35.jsonl \
#     --column_name instruction \
#     --max_samples 5000 \
#     --max_parallel_requests 16

# $143 for gen
# python scripts/data_generation/generate_responses.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2_respondedgpt4.jsonl \
#     --system_field translated_system_prompt \
#     --instruction_field translated_instruction \
#     --response_field response_on_translated_prompt \
#     --model gpt-4-0314 \
#     --temperature 0 \
#     --max_parallel_requests 4

# python scripts/data_processing/postprocess_data.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr_chat.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr_chat_postprocessed.jsonl \
#     --min_response_length 8 \
#     --embedding_model_name_or_path dangvantuan/sentence-camembert-base \
#     --threshold_similarity 0.95 \
#     --num_proc 1

# python scripts/data_processing/process_translated_data.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35_processed.jsonl \
#     --src_column query \
#     --tgt_column translated_query

# 16.18
# start from 212.26
# 2048 max tokens
# python scripts/data_processing/translate_dataset_chatgpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/open_platypus/open_platypus_merged_instruction_frgpt35_instruction_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/collected/open_platypus/open_platypus_merged_instruction_frgpt35_instruction_cleaned_frgpt35_output.jsonl \
#     --column_name output \
#     --max_tokens 2048 \
#     --max_parallel_requests 16

# $88.35
# python scripts/data_generation/generate_responses.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_10k_evolinstructed_gpt35_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_10k_evolinstructed_gpt35_cleaned_responded_gpt4.jsonl \
#     --system_field system \
#     --instruction_field evolved_instruction \
#     --response_field response_on_evolved_instruction \
#     --model gpt-4-0314 \
#     --max_tokens 2048 \
#     --max_samples 5000 \
#     --max_parallel_requests 4

# 227.13
# python scripts/data_generation/generate_instruct.py \
#     --input_prompt_file data/generation/gen_instruct/roleplay.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct.jsonl \
#     --num_instructions_to_generate 100 \
#     --temperature 1.0 \
#     --num_examples 10 \
#     --max_parallel_requests 15

# $13.15
# $13.56
# python scripts/data_generation/generate_evol_instruct.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4.jsonl \
#     --model gpt-4-0314 \
#     --max_parallel_requests 2

# 241.25
# 264.48
# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4.jsonl \
#     --instruction_field evolved_instruction \
#     --response_field response_on_evolved_instruction \
#     --model gpt-4-0314 \
#     --max_tokens 2048 \
#     --max_parallel_requests 2

# python scripts/data_generation/generate_instruct.py \
#     --input_prompt_file data/generation/gen_instruct/brainstorming.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct.jsonl \
#     --num_instructions_to_generate 100 \
#     --temperature 1.0 \
#     --num_examples 5 \
#     --max_parallel_requests 4

# python scripts/data_generation/generate_evol_instruct.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4.jsonl \
#     --model gpt-4-1106-preview \
#     --max_parallel_requests 2

# $267.60
# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4.jsonl \
#     --instruction_field evolved_instruction \
#     --response_field response_on_evolved_instruction \
#     --model gpt-4-1106-preview \
#     --max_tokens 3600 \
#     --max_parallel_requests 2

# python scripts/data_generation/generate_evol_instruct.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2_evolinstructed_gpt4turbo.jsonl \
#     --model gpt-4-0125-preview \
#     --max_samples 20000 \
#     --max_parallel_requests 4

# python scripts/data_generation/generate_evol_instruct.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2_evolinstructed_gpt4turbo.jsonl \
#     --model mistral-medium \
#     --max_parallel_requests 4

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2_evolinstructed_gpt4turbo_cleaned_new.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2_evolinstructed_gpt4turbo_cleaned_responded_gpt4.jsonl \
#     --instruction_field evolved_instruction \
#     --response_field response_on_evolved_instruction \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 4

# exit
# python scripts/data_processing/translate_dataset_gpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train_translated_question.jsonl \
#     --column_name question \
#     --max_samples 1 \
#     --max_parallel_requests 1

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train_translated_question_translated_system.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train_translated_question_translated_system_responded_gpt4.jsonl \
#     --system_field translated_system \
#     --instruction_field translated_question \
#     --response_field response_on_translated_instruction \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# $462
# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/wikipedia_fr_chunked_256_2048.jsonl \
#     --input_prompt_file data/generation/context_qa_direct.txt \
#     --output_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_direct_responded_gpt4turbo.jsonl \
#     --id_field wikidata_id \
#     --max_samples 5000 \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 2

# 617.44
# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/wikipedia_fr_chunked_256_2048.jsonl \
#     --input_prompt_file data/generation/context_qa_reasoning.txt \
#     --output_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_reasoning_responded_gpt4turbo.jsonl \
#     --id_field wikidata_id \
#     --max_samples 5000 \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# $809
# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/wikipedia_fr_chunked_256_2048.jsonl \
#     --input_prompt_file data/generation/context_qa_limited.txt \
#     --output_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_limited_responded_gpt4turbo.jsonl \
#     --id_field wikidata_id \
#     --max_samples 2000 \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# $865
# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/wikipedia_fr_chunked_256_2048.jsonl \
#     --input_prompt_file data/generation/context_qa_irrelevant.txt \
#     --output_file /projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_irrelevant_responded_gpt4turbo.jsonl \
#     --id_field wikidata_id \
#     --max_samples 1000 \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_multi_turn_chat.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train_translated_question_translated_system_responded_gpt4_processed_chat.jsonl \
#     --input_prompt_file data/generation/multi_turn_chat.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/no_robots/tmp_no_robots_train_translated_question_translated_system_responded_gpt4_processed_chat_responded_gpt4turbo.jsonl \
#     --id_field prompt_id \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# 918
# python scripts/data_generation/generate_multi_turn_chat.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4_merged_v1_processed_chat.jsonl \
#     --input_prompt_file data/generation/multi_turn_chat_2.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4_merged_v1_processed_chat_responded_gpt4turbo.jsonl \
#     --id_field id \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_gpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/dpo/dpo_mix_7k.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/dpo/dpo_mix_7k_translated_question.jsonl \
#     --column_name question \
#     --model gpt-4-0125-preview \
#     --max_tokens 2048 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/dpo/dpo_mix_7k_translated_question_processed_responded_gpt4turbo.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/dpo/dpo_mix_7k_translated_question_processed_responded_gpt4turbo_responded_vigogne2_7b.jsonl \
#     --instruction_field translated_question \
#     --response_field response_on_translated_instruction_vigogne_2_7b \
#     --model bofenghuang/vigogne-2-7b-chat \
#     --max_tokens 512 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_gpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated.jsonl \
#     --column_name function \
#     --max_tokens 2048 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated.jsonl \
#     --input_prompt_file data/generation/function_calling_with_function.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated_responded_gpt4turbo.jsonl \
#     --id_field translated_function \
#     --context_field translated_function \
#     --model gpt-4-0125-preview \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_gpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_math_querylenmin16max1024.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_math_querylenmin16max1024_translated.jsonl \
#     --column_name query \
#     --max_tokens 2048 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_math_querylenmin16max1024_translated_processed.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_math_querylenmin16max1024_translated_processed_responded_gpt4turbo0125.jsonl \
#     --instruction_field translated_query \
#     --response_field response_on_translated_prompt \
#     --model gpt-4-0125-preview \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_gpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated.jsonl \
#     --column_name question \
#     --max_tokens 2048 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated_processed.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated_processed_responded_gpt4turbo0125.jsonl \
#     --instruction_field translated_question \
#     --response_field response_on_translated_prompt \
#     --model gpt-4-0125-preview \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_by_api.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop_titled.jsonl \
#     --column_name documents \
#     --max_tokens 64 \
#     --max_parallel_requests 16

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_responded_gpt4turbo0409.jsonl \
#     --system_field system_prompt \
#     --instruction_field instruction \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# 635.46
# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop_titled_processed.jsonl \
#     --input_prompt_file data/generation/context_qa_multihop_en.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop_titled_processed_responded_gpt4turbo0409.jsonl \
#     --id_field wikidata_id \
#     --context_field documents \
#     --max_samples 5000 \
#     --model gpt-4-turbo-2024-04-09 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop_titled_processed_part2_shuffled.jsonl \
#     --input_prompt_file data/generation/context_qa_multihop_en.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_chunkedv2_min256max1024_contextqa_multihop_titled_processed_part2_1_shuffled_requested.jsonl \
#     --id_field wikidata_id \
#     --context_field documents \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0.7 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/translate_dataset_by_api.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_merged_contextqa_direct_reasoning_limited_irrlevant_multihop_responded.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_merged_contextqa_direct_reasoning_limited_irrlevant_multihop_responded_titled.jsonl \
#     --column_name documents \
#     --temperature 0.7 \
#     --max_tokens 64 \
#     --max_parallel_requests 16

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_part2.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_part2_requested.jsonl \
#     --system_field system_prompt \
#     --instruction_field instruction \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_multi_turn_chat.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4turbo_processed_chat.jsonl \
#     --input_prompt_file data/generation/multi_turn_continuation.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4turbo_processed_chat_requested.jsonl \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0.7 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_processing/prep_oasst.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/oasst/2023-11-05_oasst2_ready.trees.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/collected/oasst/oasst2_20231105_top1_en_fr_es_de_it.jsonl \
#     --top_k 1 \
#     --lang "'en,fr,es,de,it'"

# python scripts/data_processing/prep_sharegpt.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned.json \
#     --output_file /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_processed.json \
#     --validated_languages "en,fr,es,de,it"  \
#     --only_uncensored true  \
#     --num_workers 32

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106.jsonl \
#     --system_field translated_system_message \
#     --instruction_field translated_instruction \
#     --response_field response_on_translated_instruction_b \
#     --model gpt-4-1106-preview \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_samples 15000 \
#     --max_parallel_requests 6

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_maxinstlen512.jsonl \
#     --input_prompt_file data/generation/grade_prompt.txt \
#     --output_file /projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_maxinstlen512_responded.jsonl \
#     --id_field conversation_hash \
#     --context_field instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 16

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8.jsonl \
#     --input_prompt_file data/generation/grade_prompt.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluated.jsonl \
#     --id_field evolved_instruction \
#     --context_field evolved_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 16

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered_generatedinstructions_merged_maxsim09.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered_generatedinstructions_merged_maxsim09_requested.jsonl \
#     --system_field system_prompt \
#     --instruction_field instruction \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0.7 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/collected/camel/camel_merged_chemistry_physics_biology_uncensored_deduped_maxsim09_translatedmixtral8x22b_processed.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/collected/camel/camel_merged_chemistry_physics_biology_uncensored_deduped_maxsim09_translatedmixtral8x22b_processed_requested.jsonl \
#     --system_field system_prompt \
#     --instruction_field translated_instruction \
#     --model gpt-4-turbo-2024-04-09 \
#     --temperature 0 \
#     --max_tokens 4096 \
#     --max_parallel_requests 8

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed.jsonl \
#     --input_prompt_file data/generation/grade_prompt_b.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b.jsonl \
#     --id_field translated_instruction \
#     --context_field translated_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 32

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed.jsonl \
#     --input_prompt_file data/generation/grade_prompt_b.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b.jsonl \
#     --id_field translated_instruction \
#     --context_field translated_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 32

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8.jsonl \
#     --input_prompt_file data/generation/grade_prompt_b.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b.jsonl \
#     --id_field evolved_instruction \
#     --context_field evolved_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 32

# python scripts/data_generation/generate_multi_turn_response.py \
#     --input_file /projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_merged_striplast.jsonl \
#     --output_file /projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_merged_striplast_requested.jsonl \
#     --model gpt-4o-2024-05-13 \
#     --max_tokens 4096

# -- grade instruct

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8.jsonl \
#     --input_prompt_file data/generation/grade_prompt_c.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b2.jsonl \
#     --id_field evolved_instruction \
#     --context_field evolved_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 256

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed.jsonl \
#     --input_prompt_file data/generation/grade_prompt_c.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b2.jsonl \
#     --id_field translated_instruction \
#     --context_field translated_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 256

# python scripts/data_generation/generate_context_qa.py \
#     --input_document_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed.jsonl \
#     --input_prompt_file data/generation/grade_prompt_c.txt \
#     --output_file /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b2.jsonl \
#     --id_field translated_instruction \
#     --context_field translated_instruction \
#     --model meta-llama-3-70b-instruct \
#     --temperature 0 \
#     --max_tokens 1024 \
#     --max_parallel_requests 256


# input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_filtered_graded_filtered_part1_evolved_processed.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_filtered_graded_filtered_part1_evolved_processed_05.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_filtered_graded_filtered_part2_part2.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/magpie-fr-new/magpie_inst_resp_mininstscore28_head50k_1.jsonl"
input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/magpie-fr-new/magpie_inst_resp_mininstscore28_head50k_2.jsonl"

output_file="${input_file%.*}_requested.jsonl"

python scripts/data_generation/generate_response.py \
    --input_file $input_file \
    --output_file $output_file \
    --system_field system_prompt \
    --instruction_field instruction \
    --model gpt-4o-2024-11-20 \
    --max_tokens 4096 \
    --max_parallel_requests 64

# todo: sim 0.95 not 0.9
# todo: unceonsor
# todo: preprocess in training code
# todo: each eda
# todo: en parallel, circullum