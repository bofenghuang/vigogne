# Training

## Supervised Fine-tuning

For efficient LLM fine-tuning, we utilize a technique called [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) from 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. This approach involves freezing the base model's weights and introducing a small number of learnable parameters.

Additionally, for practitioners without access to GPUs with ample memory, it's advisable to consider quantizing certain computations to either 8-bit or 4-bit precision using [LLM.int8()](https://arxiv.org/abs/2208.07339) or [QLoRA](https://arxiv.org/abs/2305.14314). Be aware that this might lead to a minor reduction in speed compared to fp16 or bf16 versions.

We highly recommend the utilization of tools such as [DeepSpeed](https://github.com/microsoft/DeepSpeed) or [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api), particularly when engaged in distributed learning scenarios. When dealing with long sequences, [FlashAttention](https://arxiv.org/abs/2307.08691) becomes crucial to speed up training and minimize memory usage.

More examples can be found in [examples](https://github.com/bofenghuang/vigogne/blob/main/examples/train).

The following command shows how to fine-tune the Llama 2 7B model on a single GPU using LoRA and LLM.int8().

```bash
python vigogne/train/train_sft.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --train_file "/path/to/train/instruct/file.jsonl" \
    --output_dir "outputs/llama-2-7b-sft-instruct-lora-int8" \
    --overwrite_output_dir \
    --mode "instruct" \
    --preprocessing_num_workers "8" \
    --dataloader_num_workers "1" \
    --pack_into_block \
    --block_size "2048" \
    --load_in_8bit \
    --lora_r "64" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --per_device_train_batch_size "8" \
    --per_device_eval_batch_size "4" \
    --num_train_epochs "3" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0" \
    --torch_compile \
    --fp16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --log_level "info" \
    --logging_steps "10" \
    --logging_first_step true \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "3" \
    --report_to "tensorboard" "wandb" \
    --do_train
```
