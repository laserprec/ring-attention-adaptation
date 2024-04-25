import os

import click
import torch
import torch.distributed as dist
from configuration_llama import LlamaConfig
from modeling_llama import LlamaForCausalLM
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from sample import sample_from_logitsV1, sample_from_logitsV2
from tokenization_llama_fast import LlamaTokenizerFast
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer)
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers

# NOTE: to run this code:
# 1. cd research/ring-flash-attention && pip install -e .
# 2. cd research/ring-attention-cuda-mode/ring-llama
# 3. torchrun --nproc_per_node 2 decoding.py --prompt "What does 1+1=" --max-tokens 100 --quantized


# 1111111| 1111123| 1234343 | 1234343 
# (100000/n)^2 <<< (100000)^2

def load_model(
    model_name: str,
    cache_dir: str,
    torch_dtype: torch.dtype,
    device: torch.DeviceObjType,
    skip_load: bool = False,
    no_weight_init: bool = False,
    is_quantized: bool = False,
):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    if skip_load:
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        config._attn_implementation = "flash_attention_2"
        config = LlamaForCausalLM._autoset_attn_implementation(
            config, torch_dtype=torch_dtype
        )
        print("using llama config:", config)
        init_contexts = [no_init_weights(_enable=no_weight_init)]
        with ContextManagers(init_contexts):
            LlamaForCausalLM._set_default_torch_dtype(torch_dtype)
            model = LlamaForCausalLM(config)
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map=device,
            load_in_4bit=True if is_quantized else False,
        )
    return model, tokenizer


# source https://stackoverflow.com/a/1094933
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def add_padding(input_chunks, tokenizer, world_size):
    pad_tensor = torch.tensor(
        [int(tokenizer.pad_token_id)]
        * (
            input_chunks.shape[1] % (world_size)
        ),
    dtype=int).unsqueeze(0)
    print("size of pad_tensor", pad_tensor.shape)
    return torch.cat([pad_tensor.int(), input_chunks.int()], dim=1)

def remove_padding(input_chunks, tokenizer):
    # Assuming padding is at the start and all padding tokens are identical
    pad_token_id = int(tokenizer.pad_token_id)

    # Find the first non-pad token in each example in the batch
    # We look across dim=1 (columns) to find the first non-pad token
    # `all(dim=0)` ensures we do not remove columns prematurely across batch elements if they're padded unevenly
    if input_chunks.size(1) > 0 and torch.all(input_chunks == pad_token_id, dim=0).any():
        # Get the index of the first column where not all entries are the pad_token_id
        first_non_pad_index = (input_chunks != pad_token_id).any(dim=0).nonzero(as_tuple=True)[0][0]
        # Slice from the first non-pad token to the end
        input_chunks = input_chunks[:, first_non_pad_index:]
    else:
        # No padding detected or empty tensor, return as is
        return input_chunks

    print("Size of tensor after removing padding:", input_chunks.shape)
    return input_chunks

def generate(
    model,
    tokenizer,
    world_size,
    rank,
    device,
    prompt,
    max_new_tokens=50,
    temperature=0.9,
    top_k=5,
):
    tokenized_input = tokenizer(prompt, return_tensors="pt").input_ids
    for i in range(max_new_tokens):
        print("size of input", tokenized_input.shape)
        tokenized_input = add_padding(tokenized_input, tokenizer, world_size)
        position_ids = torch.arange(tokenized_input.shape[1]).unsqueeze(0)
        # break it up into chunks
        input_chunks = tokenized_input.chunk(chunks=world_size, dim=1)[rank]
        position_ids = position_ids.chunk(chunks=world_size, dim=1)[rank]
        print("size of org input chunk", input_chunks.shape)
        input_chunks = input_chunks.to(device)
        position_ids = position_ids.to(device)

        print("size of input chunk", input_chunks.shape)
        y = model(input_ids=input_chunks, position_ids=position_ids).logits
        print("size of output chunk", y.shape)
        gathered_logits = [torch.zeros_like(y) for _ in range(world_size)]
        #     1         2       0
        # [ XXXXXXX | YYYYYY | ZZZZ[?] ]
        torch.distributed.all_gather(gathered_logits, y, group=None, async_op=False)
        torch.distributed.barrier()

        concat_logits = torch.cat(gathered_logits, dim=1).squeeze()
        sampled_logits = sample_from_logitsV1(concat_logits, strategy="greedy")
        print("size of output logits",sampled_logits.shape)
        predicted_ids = sampled_logits[-1]
        print("output predicted_id", predicted_ids, "tokenizer.eos_token_id: ", tokenizer.eos_token_id)
        if predicted_ids.int() == tokenizer.eos_token_id:
            print("Hit EOS token. Stop generation")
            break
        decoded_text = tokenizer.decode([predicted_ids])
        print(f"decoded_text for rank: {rank}:", " ".join(decoded_text))
        tokenized_input = torch.cat(
            [tokenized_input.int(), torch.Tensor([predicted_ids]).unsqueeze(0).int()], dim=1
        )
        tokenized_input = remove_padding(tokenized_input, tokenizer)
    return tokenized_input


@torch.inference_mode()
@click.command()
@click.option("--prompt", default="What's 1+1=", help="The prompt text to display.", type=str)
@click.option("--max-tokens", default=100, help="The max token to generate", type=int)
@click.option('--quantized', is_flag=True, help="Whether to 4-bit quantize the model")
def main(prompt, max_tokens, quantized):
    dtype = torch.float16
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:0")
    print(f"world_size: {world_size}, device: {device}")

    skip_load = False
    model, tokenizer = load_model(
        # "/efs/jianjl/playground/coding/v2_workspace/llama_v2_epoch8",
        # cache_dir="/efs/jianjl/playground/coding/v2_workspace/",
        "/efs/jianjl/playground/qa/llama2_pretrained",
        cache_dir="/efs/jianjl/playground/qa/",
        torch_dtype=dtype,
        device=device,
        skip_load=skip_load,
        no_weight_init=skip_load,
        is_quantized=quantized,
    )

    # TODO: parametrize the prompt?
    # prompt = "Hello who are you? Hey there"
    # prompt = "what does 1+1="
    # prompt = "what is post-term pregnancy"

    torch.cuda.reset_peak_memory_stats()
    max_mem_allocated_before = torch.cuda.max_memory_allocated(device)
    tokens_generated = generate(
        model,
        tokenizer,
        world_size,
        rank,
        device,
        prompt,
        max_new_tokens=max_tokens,
    )
    decoded_text = tokenizer.batch_decode(sequences=tokens_generated)
    if rank == 0:
        print("*"*10)
        print("Prompt:", prompt)
        print(f"Generated text ({rank} rank):", decoded_text)
        print("*"*10)

if __name__ == "__main__":
    main()
