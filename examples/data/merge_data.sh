#!/usr/bin/env bash

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export BITSANDBYTES_NOWELCOME="1"
# export CUDA_VISIBLE_DEVICES="4"

# V1
################################################################################
# instruct data log
# python vigogne/data/merge_datasets.py --inputs-files /projects/bhuang/corpus/text/llm/alpaca/alpaca_data_cleaned_fr_52k_train.jsonl /projects/bhuang/corpus/text/llm/self_instruct/self_instruct_data_final_100k_new.jsonl data/instruct/dolly_bactrian_fr_15k.jsonl --output-file data/instruct/tmp_alpaca_selfinstruct_dollybactrian.jsonl
# python vigogne/data/merge_datasets.py --inputs-files data/instruct/tmp_alpaca_selfinstruct_dollybactrian.jsonl /projects/bhuang/corpus/text/llm/grade_school_math_instructions/grade_school_math_instructions.jsonl /projects/bhuang/corpus/text/llm/logic_inference_oa/logic_inference_oa_10k.jsonl /projects/bhuang/corpus/text/llm/hc3/hc3_dedup_10k.jsonl /projects/bhuang/corpus/text/llm/code_alpaca/code_alpaca_20k.jsonl /projects/bhuang/corpus/text/llm/evol_instruct/evol_instruct_143k_max512_50k.jsonl --output-file data/instruct/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl

# chat data log
# python vigogne/data/convert_alpaca_to_chat.py data/instruct/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl
# python vigogne/data/convert_alpaca_to_chat.py data/instruct/tmp_alpaca_data_cleaned_fr_52k_test.jsonl data/chat/tmp_alpaca_data_cleaned_fr_52k_test.jsonl
# python vigogne/data/merge_datasets.py --inputs-files data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct.jsonl /projects/bhuang/corpus/text/llm/self_chat/self_chat_data_quora_fr_50k.jsonl /projects/bhuang/corpus/text/llm/oasst/oasst_20230412_fr_top1.jsonl /projects/bhuang/corpus/text/llm/oasst/oasst_20230412_en_top1.jsonl /projects/bhuang/corpus/text/llm/sharegpt_90k/sg_90k_all_cleaned_fr.jsonl /projects/bhuang/corpus/text/llm/sharegpt_90k/sg_90k_all_cleaned_en.jsonl /home/bhuang/nlp/vigogne/data/chat/dummy_chat.jsonl /projects/bhuang/corpus/text/llm/baize/alpaca_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/medical_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/quora_chat_data_10k.jsonl /projects/bhuang/corpus/text/llm/baize/stackoverflow_chat_data_20k.jsonl --output-file data/chat/tmp_alpaca_selfinstruct_dollybactrian_gradeschoolmath_logicinference_hc3_codealpaca_evolinstruct_selfchatquora_oasstfren_sgfren_dummy_baize.jsonl


# V2
################################################################################
# instruction data
# 1st stage
# alpaca 50k: /projects/bhuang/corpus/text/llm/generated/translated_alpaca/alpaca_data_cleaned_fr_52k_train_v2.jsonl
# self-instruct 100k: /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_v2.jsonl
# dolly bactrian 15k: /projects/bhuang/corpus/text/llm/collected/dolly/dolly_bactrian_fr_15k_v2.jsonl
# orca gpt3.5 gen 100k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_fr.jsonl
# orca gpt3.5 parallel 100k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_en.jsonl
# gpt4llm 52k: /projects/bhuang/corpus/text/llm/collected/gpt4_llm/alpaca_gpt4_data_v2.jsonl
# evol instruct 100k: /projects/bhuang/corpus/text/llm/collected/evol_instruct/evol_instruct_143k_v2_max2048_100k.jsonl
# codealpaca 20k: /projects/bhuang/corpus/text/llm/collected/code_alpaca/code_alpaca_20k_v2.jsonl
# 2nd stage
# orca gpt4 gen 20k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr.jsonl
# orca gpt4 parallel 20k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_en.jsonl
# airoboros 20k: /projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_gpt4_1.4_max2048_20k.jsonl
# lima 798: /projects/bhuang/corpus/text/llm/collected/lima/lima_instruct_max1024.jsonl

# python scripts/data_processing/merge_datasets.py -i /projects/bhuang/corpus/text/llm/generated/translated_alpaca/alpaca_data_cleaned_fr_52k_train_v2.jsonl /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_v2.jsonl /projects/bhuang/corpus/text/llm/collected/dolly/dolly_bactrian_fr_15k_v2.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_fr.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_en.jsonl /projects/bhuang/corpus/text/llm/collected/gpt4_llm/alpaca_gpt4_data_v2.jsonl /projects/bhuang/corpus/text/llm/collected/evol_instruct/evol_instruct_143k_v2_max2048_100k.jsonl /projects/bhuang/corpus/text/llm/collected/code_alpaca/code_alpaca_20k_v2.jsonl -o /projects/bhuang/corpus/text/llm/final/v2/merged_alpacafr_selfinstructfr_dollyfr_orcagpt35fr_orcagpt35en_gpt4llmen_evolinstructen_codealpaca.jsonl
# python scripts/data_processing/merge_datasets.py -i /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_en.jsonl /projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_gpt4_1.4_max2048_20k.jsonl /projects/bhuang/corpus/text/llm/collected/lima/lima_instruct_max1024.jsonl -o /projects/bhuang/corpus/text/llm/final/v2/merged_orcafr_orcaen_airoborosen_limaen.jsonl


# Vigogne-Chat 2.0
################################################################################

# 1st stage
# fr 177k, en 280k
# instruction data
# alpaca bactrian uncensored 47k: /projects/bhuang/corpus/text/llm/collected/alpaca_bactrian_fr/alpaca_bactrian_fr_uncensored_chat.jsonl
# dolly bactrian uncensored 13k: /projects/bhuang/corpus/text/llm/collected/dolly/dolly_bactrian_fr_15k_uncensored_chat.jsonl
# orca gpt3.5 gen 115k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_fr_chat.jsonl
# orca gpt3.5 parallel 115k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_en_chat.jsonl
# gpt4llm uncensored 45k: /projects/bhuang/corpus/text/llm/collected/gpt4_llm/alpaca_gpt4_data_unfiltered_chat.jsonl
# evol instruct uncensored 50k: /projects/bhuang/corpus/text/llm/collected/evol_instruct/evol_instruct_143k_uncensored_chat_max2048_50k.jsonl
# codealpaca 20k: /projects/bhuang/corpus/text/llm/collected/code_alpaca/code_alpaca_20k_chat.jsonl
# chat data
# sg max2048 1st fr 1642: /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_splitted2048_fr.jsonl
# sg max2048 1st en 50k: /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_splitted2048_en.jsonl
# oasst top1 fr 256: /projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_fr_top1_v3.jsonl
# oasst top1 en 3783: /projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_en_top1_v3.jsonl
# mt data

# 2nd stage
# instruction data
# orca gpt4 gen 20k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr_chat.jsonl
# orca gpt4 parallel 20k: /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_en_chat.jsonl
# airoboros 20k: /projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_gpt4_1.4_max2048_20k_chat.jsonl
# lima 956: /projects/bhuang/corpus/text/llm/collected/lima/lima_chat_max2048.jsonl

# 406883
# python scripts/data_processing/merge_datasets.py -i /projects/bhuang/corpus/text/llm/collected/alpaca_bactrian_fr/alpaca_bactrian_fr_uncensored_chat.jsonl /projects/bhuang/corpus/text/llm/collected/dolly/dolly_bactrian_fr_15k_uncensored_chat.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_fr_chat.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_en_chat.jsonl /projects/bhuang/corpus/text/llm/collected/gpt4_llm/alpaca_gpt4_data_unfiltered_chat.jsonl /projects/bhuang/corpus/text/llm/collected/evol_instruct/evol_instruct_143k_uncensored_chat_max2048_50k.jsonl /projects/bhuang/corpus/text/llm/collected/code_alpaca/code_alpaca_20k_chat.jsonl -o /projects/bhuang/corpus/text/llm/final/v3/merged_alpacabactrianfr_dollybactrianfr_orcagpt35fr_orcagpt35parallelen_gpt4llmen_evolinstructen_codealpaca.jsonl
# 463450
# python scripts/data_processing/merge_datasets.py -i /projects/bhuang/corpus/text/llm/final/v3/merged_alpacabactrianfr_dollybactrianfr_orcagpt35fr_orcagpt35parallelen_gpt4llmen_evolinstructen_codealpaca.jsonl /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_splitted2048_fr.jsonl /projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_splitted2048_en.jsonl /projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_fr_top1_v3.jsonl /projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_en_top1_v3.jsonl -o /projects/bhuang/corpus/text/llm/final/v3/merged_alpacabactrianfr_dollybactrianfr_orcagpt35fr_orcagpt35parallelen_gpt4llmen_evolinstructen_codealpaca_sgfr_sgen_oasstfr_oassten.jsonl
# 60956
# python scripts/data_processing/merge_datasets.py -i /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_fr_chat.jsonl /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_en_chat.jsonl /projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_gpt4_1.4_max2048_20k_chat.jsonl /projects/bhuang/corpus/text/llm/collected/lima/lima_chat_max2048.jsonl -o /projects/bhuang/corpus/text/llm/final/v3/merged_orcagpt4fr_orcagpt4parallelen_airoborosen_limaen.jsonl


# Vigogne-Chat 2.1 (tmp fast version)
################################################################################

# Flan-V2 Filtered-Translation Filtered-Gen
# Flan-V2 Translated-Fr-M2M100 Gen-Orca-GPT4 Filtered-Length-Heuristic-Sim0.9 20k
# Flan-V2 Parallel-En Gen-Orca-GPT4 20k

# Flan-V2-COT Translated-Fr-GPT3.5 Gen-Orca-GPT4 Filtered-Length-Heuristic-Sim0.9 5k
# Flan-V2-COT Parallel-En Gen-Orca-GPT4 5k

# Meta-Math-QA Translated-Fr-GPT3.5 Gen-GPT4 5k
# Evol-Instruct-Code Translated-Fr-GPT3.5 Gen-GPT4 5k

# SG Split2048-1st Fr Length2048 Sim0.9 1642
# Dove Length-2048 En Length2048 Sim0.9 3007
# OASST Top1 Fr Length2048 Sim0.9 256
# OASST Top1 En Length2048 Sim0.9 3420

# Airoboros-2.2.1 Length-2048 Sim-0.95 Sampled-Longest 10k
# Lima Length-? 956

# todo: math
# Megacode-Best Length-2048 Sim-0.95 Sampled-Longest 20k

# todo: longest, Filtered-Length-Heuristic-Sim
# Flan-V2 Translated-Fr-M2M100 Gen-Orca-GPT3.5 Sampled 40k


# python scripts/data_processing/merge_datasets.py \
#     -i \
#     "/projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_processed2_en.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_processed2_fr.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/orca/part_2_processed/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35_normalized_respondedgpt4_processed_en.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/orca/part_2_processed/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35_normalized_respondedgpt4_processed_fr.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35_processed_prompted_respondedgpt4_fr.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2_respondedgpt4_processed_fr.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/sharegpt_90k/sg_90k_all_cleaned_splitted2048_fr_processed.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/dove/pure_dove_max2048_maxsim09.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_fr_top1_max2048_maxsim09.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/oasst/oasst_20230412_en_top1_max2048_maxsim09.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_2.2.1_max2048_maxsim095_sampled.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/lima/lima_chat_max2048_maxsim095.jsonl" \
#     "/projects/bhuang/corpus/text/llm/collected/megacode-best/megacodebest_max2048_maxsim095_sampled.jsonl" \
#     "/projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed2_sampled40k_fr.jsonl" \
#     -o \
#     "/projects/bhuang/corpus/text/llm/final/v4/all.jsonl"


# V3.0
################################################################################

# self-instruct 46k
# self-instruct-v2.1 filtered-sim0.8 evol-instruct-gpt-3.5t responded-gpt4-0314 4458
# self-instruct-v2.2 filtered-sim0.8 evol-instruct-mistral-medium responded-gpt4t-0125 42207

# wild-chat 20k
# wild-chat-1m filtered prompt-evaluated-clustered-regenerated-deduped0.9 responded-gpt4t-20240409 20220

# open-orca 36k
# open-orca-v1 translated-m2m100-1b2 filtered-length-heuristic-sim0.9 responded-gpt4-0314 18256
# open-orca-v2-cot translated-gpt-3.5t filtered-length-heuristic-sim0.9 responded-gpt4-0314 4635
# open-orca-v3 translated-gpt-3.5t filtered-length-heuristic-sim0.9 responded-gpt4t-1106 14351

# math 40k
# meta-math-qa-v0.1 translated-gpt-3.5t responded-gpt4-0314 4673
# meta-math-qa-v0.2-math filtered-length translated-gpt-3.5t responded-gpt4t-0125 5031
# orca-math filtered-sim0.8 translated-gpt-3.5t responded-gpt4t-0125 20087
# math-instruct-cot-cross downsampled-camel-math filtered-prev-sim0.9 filtered-sim0.9 responded-gpt4t-20240409 9565

# code 4k
# evol-instruct-code translated-gpt-3.5t responded-gpt4-0314 4383

# stem
# camel-chemistry/physics/biology 6,171

# roleplay+brainstorming multi-turn 2k
# roleplay filtered-sim0.7 evol-instructed-gpt-4-0314 responded-gpt-4-0314 multiturn-gpt4t-0125 1343
# brainstorming filtered-sim0.7 evol-instructed responded-gpt4t-1106 multiturn-gpt4t-0125 708

# no-robots 10k
# no-robots single/multi-turn translated-google responded-gpt4-turbo-0125 8700+744

# context-qa 23k
# context-qa wikipedia-fr direct/reasoning/limited/irrelevant/multihop responded-gpt4-turbo-0125/0409 23000

# function-calling 3k
# glaive-fc filtered-length-512 dedup-sim-0.95 translated-gpt-3.5t generated 3701

# self

# general
# lima 1030
# oasst2-20231105 top1 en-fr-es-de-it 10746 (5419, 4258, 572, 350, 147)
# airoboros 52k
# capybara 16k

# wild 30k
# sharegpt4-openchat 4k (4479 en + 138 fr)
# lmsys-chat filtered-gpt4 6k
# wild-chat filtered-gpt4 43k
# deita

# code 49k
# code-feedback 15k
# code-feedback-filtered-instruction (single-turn) 33k

# agent
# agent-instruct 1515


# ---
# fr

# self-instruct
# /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_10k_evolinstructed_gpt35_cleaned_responded_gpt4_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_data_final_100k_new_merged_instruction_cleaned_sim08v2_evolinstructed_gpt4turbo_cleaned_responded_gpt4turbo_processed_chat.jsonl
# merged-promptevaluated
# /projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b_minscore2.jsonl

# wild-chat
# /projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered_generatedinstructionmixtral8x22b_merged_maxsim09_requested_responded_batch_gpt4turbo0409_processed.jsonl

# open-orca
# /projects/bhuang/corpus/text/llm/generated/orca/part_1_processed/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt4_processed2_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/orca/part_2_processed/1m_gpt4_augmented_deduped_lot2_questionmax1024min64_cot_sim095_frgpt35_normalized_respondedgpt4_processed_chat.jsonl
# merged-v1-v2-promptevaluated, v3-promptevaluated
# /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b_processed.jsonl

# math
# /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_questionMin16Max1024_sim095_frgpt35_processed_prompted_respondedgpt4_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_math_querylenmin16max1024_translated_processed_responded_gpt4turbo0125_processed_chat.jsonl
# merged-v1-v2
# /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_merged_v01_v02_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated_processed_responded_gpt4turbo0125_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_responded_gpt4turbo0409_processed_chat_merged_processed.jsonl

# code
# /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2_respondedgpt4_processed_chat.jsonl

# stem
/projects/bhuang/corpus/text/llm/collected/camel/camel_merged_chemistry_physics_biology_uncensored_deduped_maxsim09_translatedmixtral8x22b_processed_requested_responded_batch_gpt4turbo0409_processed.jsonl

# roleplay+brainstorming multi-turn
# /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4_merged_v1_processed_chat_responded_gpt4turbo_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4turbo_processed_chat_requested_responded_batch_gpt4turbo0409_processed_chat.jsonl

# no-robots
# /projects/bhuang/corpus/text/llm/generated/no_robots/no_robots_train_translated_question_translated_system_responded_gpt4_processed_chat.jsonl

# context-qa
# /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_merged_contextqa_direct_reasoning_limited_irrlevant_multihop_responded_titled_processed2_chat.jsonl

# function-calling
# /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated_responded_gpt4turbo_processed_chat.jsonl

/projects/bhuang/corpus/text/llm/collected/lima/lima_processed.jsonl
/projects/bhuang/corpus/text/llm/collected/oasst/oasst2_20231105_top1_en_fr_es_de_it.jsonl
/projects/bhuang/corpus/text/llm/collected/airoboros/airoboros_3.2_processed_no_system_no_coding_unalign.jsonl
/projects/bhuang/corpus/text/llm/collected/capybara/capybara_16k_processed.jsonl
/projects/bhuang/corpus/text/llm/collected/sharegpt4_openchat/sharegpt4openchat_lmsyschat1m_wildchat_deita_processed_en_fr_es_de_it_maxsim09_uncensored.jsonl
/projects/bhuang/corpus/text/llm/collected/code_feedback/code_feedback_totmaxlen4000_instmaxsim08_uncensored.jsonl
/projects/bhuang/corpus/text/llm/collected/code_feedback/code_feedback_filtered_instruction_majorlanguages_instmaxlen1024_respmaxlen2048_instmaxsim08_uncensored.jsonl
/projects/bhuang/corpus/text/llm/collected/agent_instruct/agent_instruct_processed.jsonl

# ---

/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b_minscore2.jsonl
/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered_generatedinstructionmixtral8x22b_merged_maxsim09_requested_responded_batch_gpt4turbo0409_processed.jsonl
/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b_processed.jsonl
/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/metaMathQA_merged_v01_v02_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/orca_math_word_problems_maxsim08_translated_processed_responded_gpt4turbo0125_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/math/mathinstruct_cot_metamathorcamaxsim09_maxsim08_processed_responded_gpt4turbo0409_processed_chat_merged_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/code/evol_instruct_code_80k_v1_questionmin64max1024_sim09_frgpt35_processed2_respondedgpt4_processed_chat.jsonl
/projects/bhuang/corpus/text/llm/collected/camel/camel_merged_chemistry_physics_biology_uncensored_deduped_maxsim09_translatedmixtral8x22b_processed_requested_responded_batch_gpt4turbo0409_processed.jsonl
# /projects/bhuang/corpus/text/llm/generated/roleplay/roleplay_v3_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4_merged_v1_processed_chat_responded_gpt4turbo_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/brainstorming/brainstorming_v1_instruct_cleaned_evolved_gpt4_cleaned_responded_gpt4turbo_processed_chat_requested_responded_batch_gpt4turbo0409_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/no_robots/no_robots_train_translated_question_translated_system_responded_gpt4_processed_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/context_qa/wikipedia_fr_merged_contextqa_direct_reasoning_limited_irrlevant_multihop_responded_titled_processed2_chat.jsonl
# /projects/bhuang/corpus/text/llm/generated/function_calling/glaive_fc_function_maxlen512_dedup_maxsim95_translated_responded_gpt4turbo_processed_chat.jsonl

/projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_filteredempty_maxtotlen8192.jsonl

# 119,580
# /projects/bhuang/corpus/text/llm/merged/v3_1/sharegpt4openchat_lmsyschat1m_wildchat_deita_lima_oasst2_airoboros_capybara_codefeedback_codefeedbackfilteredinstruction_agentinstruct_merged_nonemptyturn_totminlen64_instmaxsim095.jsonl
# 89,292
/projects/bhuang/corpus/text/llm/merged/v3_1/sharegpt4openchat_deita_lima_airoboros_capybara_codefeedback_codefeedbackfilteredinstruction_merged_nonemptyturn_totminlen64_instmaxsim095.jsonl

