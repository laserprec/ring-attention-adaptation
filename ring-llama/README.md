# ring-llama

Experiments with ring-attention & llama (WIP).

Current state: Scaffolding for future experiments.
Currently allows to iterate quickly with llama 7b (optionally skip initialization).

For example decoding (greedy):
Run `torchrun --nproc_per_node 2 decoding.py --prompt "What does 1+1=" --max-tokens 100 --quantized`

For exploration of how much longer context window ring attention can fit with forward pass:
Run `torchrun --nproc_per_node 4 main.py --context_window 32000 --quantized`

