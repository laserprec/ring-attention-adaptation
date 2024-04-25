import os
import click
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from configuration_llama import LlamaConfig
from modeling_llama import LlamaForCausalLM
from tokenization_llama_fast import LlamaTokenizerFast
import torch.distributed as dist
from sample import sample_from_logitsV2, sample_from_logitsV1

from ring_flash_attn import ring_flash_attn_qkvpacked_func

# NOTE: to run this code:
# 1. cd research/ring-flash-attention && pip install -e .
# 2. cd research/ring-attention-cuda-mode/ring-llama
# 3. torchrun --nproc_per_node 4 main.py --context_window 32000 --quantized

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

@torch.inference_mode()
@click.command()
@click.option('--context_window', default=10000, help='The size of the context window.', type=int)
@click.option('--quantized', is_flag=True, help="Whether to 4-bit quantize the model")
def main(context_window, quantized):
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
        "/efs/jianjl/playground/qa/llama2_pretrained",
        cache_dir="/efs/jianjl/playground/qa/",
        torch_dtype=dtype,
        device=device,
        skip_load=skip_load,
        no_weight_init=skip_load,
        is_quantized=quantized,
    )
    torch.cuda.reset_peak_memory_stats()
    max_mem_allocated_before = torch.cuda.max_memory_allocated(device)

    # temporarily use dummy input (ensure shape is same for both devices)
    length = 20000  # Set this to your desired length
    mod_value = 32000  # Set the upper limit for values (llama2 vocab size is 32k)
    tokenized_input = torch.arange(length) % mod_value
    tokenized_input = tokenized_input.unsqueeze(0)  # To match your original shape requirement
    position_ids = torch.arange(length) % mod_value
    position_ids = position_ids.unsqueeze(0)  # To match your original shape requirement

    print("tokenized_input", tokenized_input.shape)
    print("position_ids", position_ids.shape)

    input_chunks = tokenized_input.chunk(chunks=world_size, dim=1)
    position_ids = position_ids.chunk(chunks=world_size, dim=1)

    x = input_chunks[rank]
    x_pos_ids = position_ids[rank]

    x = x.to(device)
    x_pos_ids = x_pos_ids.to(device)
    print(f"model input x for rank: {rank}: {x.shape} (position_ids: {x_pos_ids.shape})")

    y = model(x, position_ids=x_pos_ids).logits

    print(f"output logits for rank: {rank}:", y.shape, y.dtype, y.device)

    gathered_logits = [torch.zeros_like(y) for _ in range(world_size)]

    torch.distributed.all_gather(gathered_logits,  y, group=None, async_op=False)

    max_mem_allocated_after = torch.cuda.max_memory_allocated(device)
    print(
        f"{device} delta in meory: {sizeof_fmt(max_mem_allocated_after-max_mem_allocated_before)}\n"
    )
    concat_logits = torch.cat(gathered_logits, dim=1).squeeze()
    sampled_logits = sample_from_logitsV1(concat_logits, strategy="greedy")
    print("size of output logits",sampled_logits.shape)
    predicted_ids = sampled_logits[-1]
    print("predicted ids", predicted_ids, f"type: {type(predicted_ids)}")
    decoded_text = tokenizer.decode([predicted_ids])
    print(f"decoded_text for rank: {rank}:", "".join(decoded_text))


if __name__ == "__main__":
    main()
