import os
import torch
import torch.nn as nn
import torch.distributed as dist

import click

from transformers import LlamaTokenizerFast, AutoTokenizer
from torch.optim import AdamW
from modeling_llama import LlamaForCausalLM

# NOTE: to run this code:
# 1. cd research/ring-flash-attention && pip install -e .
# 2. cd research/ring-attention-cuda-mode/ring-llama
# 3. torchrun --nproc_per_node 4 backward.py --context_window 100 --quantized


def load_model(
    model_name: str,
    torch_dtype: torch.dtype,
    device: torch.DeviceObjType,
    is_quantized: bool = False,
):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=device,
        load_in_4bit=True if is_quantized else False,
    )
    return model, tokenizer

def create_input_position_targets(length, mod_value, device):
    tokenized_input = torch.arange(length) % mod_value
    tokenized_input = tokenized_input.unsqueeze(0)  # Add batch dimension
    
    position_ids = torch.arange(length) % mod_value
    position_ids = position_ids.unsqueeze(0)  # To match your original shape requirement
    
    # Dummy target labels (randomly generated for demonstration; replace with actual labels in practice)
    target_labels = torch.randint(0, mod_value, (1, length)).long()
    
    tokenized_input = tokenized_input.to(device)
    position_ids = position_ids.to(device)
    target_labels = target_labels.to(device)
    return tokenized_input, position_ids, target_labels

def prepare_inputs_n_labels(tokenized_input, position_ids, target_labels, world_size, rank):
    input_chunks = tokenized_input.chunk(chunks=world_size, dim=1)[rank]
    position_id_chunks = position_ids.chunk(chunks=world_size, dim=1)[rank]
    target_label_chunks = target_labels.chunk(chunks=world_size, dim=1)[rank]

    return input_chunks, position_id_chunks, target_label_chunks

@click.command()
@click.option('--context_window', default=100, help='The size of the context window.', type=int)
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

    # Configuration
    length = context_window  # Desired sequence length for the input
    mod_value = 32000  # Vocabulary size of the LLaMA2 model

    # Load model and tokenizer
    model_id = "/efs/jianjl/playground/qa/llama2_pretrained"
    model, tokenizer = load_model(
        model_id, 
        torch_dtype=dtype,
        device=device,
        is_quantized=quantized,
    )

    # Prepare tokenized input
    tokenized_input, position_ids, target_labels = create_input_position_targets(length, mod_value, device)
    if tokenized_input.shape[1] % world_size != 0:
        raise ValueError(f"Input tensor needs to be divisible by {world_size}. Current shape {tokenized_input.shape}")
    input_chunks, position_id_chunks, target_label_chunks = prepare_inputs_n_labels(tokenized_input, position_ids, target_labels, world_size, rank)
    print("input_chunk:", input_chunks.shape, "position_ids_chunk:", position_id_chunks.shape, "target_label_chunks:", target_label_chunks.shape)
    # Ensure model is in training mode
    model.train()
    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Forward pass
    outputs = model(input_chunks)
    logits = outputs.logits
    print(f"Rank {rank} forward pass done. Logits: {logits.shape}")

    # Compute loss
    loss = loss_function(logits.view(-1, mod_value), target_label_chunks.view(-1))

    # Backward pass and optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Output the loss
    print(f"Rank {rank} Loss:", loss.item())

if __name__ == "__main__":
    main()


