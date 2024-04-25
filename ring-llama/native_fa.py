import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaConfig, LlamaForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.module")

efs_path = "/efs/jianjl/playground/coding/v2_workspace/llama_v2_epoch8"

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def load_with_sharding(path, memory_map, torch_dtype=torch.bfloat16):
    auto_model_class = AutoModelForCausalLM
    with init_empty_weights():
        # model = auto_model_class.from_config(
        #     model_config,
        #     torch_dtype=torch_dtype,
        # )
        model = auto_model_class.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        # device_map actually a dictionary specifying the memory allocation for different devices
        if type(memory_map) == dict:
            device_map = infer_auto_device_map(
                model,
                max_memory=memory_map,
                no_split_module_classes=["LlamaDecoderLayer"],
            )
        else:
            raise ValueError("Incorrect memory_map type")
        # loading model weights with auto sharding
        # See huggingface docs for more details on designing the device map for sharding:
        # https://huggingface.co/docs/accelerate/v0.20.0/en/usage_guides/big_modeling#designing-a-device-map
        model = load_checkpoint_and_dispatch(
            model,
            path,
            device_map=device_map,
            no_split_module_classes=["LlamaDecoderLayer"],
            offload_folder="offload_cpu",
            offload_state_dict=True,
        )
    return model

rank=3
device = torch.device(f"cuda:{rank}")
max_mem_allocated_before = torch.cuda.max_memory_allocated(device)

tokenizer = AutoTokenizer.from_pretrained(efs_path)
model = AutoModelForCausalLM.from_pretrained(
        efs_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        load_in_4bit=True,
    )

# 8 GPUS
# memory_map = {0: "1GiB", 1: "2GiB", 2: "2GiB", 3: "2GiB", 4: "2GiB", 5: "2GiB", 6: "2GiB", 7: "2GiB", "cpu": "60GiB"}
# 4 GPUS
# memory_map = {0: "3GiB", 1: "3GiB", 2: "4GiB", 3: "4GiB", "cpu": "60GiB"}
# 2 GPUS
# memory_map = {0: "6GiB", 1: "8GiB", "cpu": "60GiB"}
# model = load_with_sharding(efs_path, memory_map = memory_map)

# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

print(model.hf_device_map)
print("Attn implementation", model.model)


max_mem_allocated_after_model_loaded = torch.cuda.max_memory_allocated(device)
print(
    f"{device} delta in meory after model loaded: {sizeof_fmt(max_mem_allocated_after_model_loaded-max_mem_allocated_before)}\n"
)
tokenized_input = torch.arange(9000).unsqueeze(0)
position_ids = torch.arange(9000).unsqueeze(0)

model.eval()
x = tokenized_input.to(device)



y = model(x, position_ids=position_ids).logits

def greedy(logits, filter_value=float("-inf")):
    probabilities = F.softmax(logits, dim=-1)    
    sampled_token = torch.argmax(probabilities, dim=-1)
    return sampled_token

print("y shape:", y.shape)
logits = greedy(y)
print("after greedy", logits.shape)
max_mem_allocated_after = torch.cuda.max_memory_allocated(device)
print(
    f"{device} delta in meory: {sizeof_fmt(max_mem_allocated_after-max_mem_allocated_before)}\n"
)
decoded_text = tokenizer.batch_decode(sequences=logits)
print(f"decoded_text for rank: {rank}:", "".join(decoded_text)[:200])