# GMT (Graph-as-Memory Tuning)

GMT is our KG-LLM fusion method that represents local graph structure as explicit memory and injects it into multiple Transformer layers via cross-attention. This enables deep, token-wise evidence retrieval during generation rather than shallow prefix concatenation.

## Key ideas
- Graph-as-Memory: compress a local subgraph into a fixed number of memory tokens.
- SGM (Semantic Graph Module): relation-centric message passing to extract semantic evidence.
- Memory Cross-Attention: fuse graph memory into multiple Transformer layers.
- Efficient adaptation: apply LoRA only to memory cross-attention, keep the base LLM frozen.

## Repository layout
- `gmt_model.py`: GMT model components (SGM, memory encoder, cross-attn integration).
- `graph_data.py`: KG indexing and data loading utilities.
- `pretrain_gmt.py`: Stage 1 SGM pretraining.
- `finetune_gmt.py`: Stage 2 memory-augmented LLM finetuning.
- `inference_gmt.py`: inference and evaluation (classification and link prediction).
- `build_rel_semantics.py`: build relation semantic vectors (optional).
- `process_kge.py`: load KGE embeddings as relation semantics (optional).
- `data/`: sample JSON data.
- `kg/`: OpenKE-style KG files.
- `templates/`, `utils/`: prompt templates and helpers.

## Requirements
Recommended: Python 3.9+.

Core dependencies:
- `torch`
- `transformers`
- `datasets`
- `sentencepiece`

Optional:
- `sentence-transformers` (only for `build_rel_semantics.py`)

## Model support
- Current implementation is LLaMA-specific (`LlamaForCausalLM` and LLaMA decoder internals).
- `--base_model` should point to a LLaMA-compatible checkpoint.

## Data format
Training and inference JSON or JSONL files must include:
- `instruction` (string)
- `input` (string)
- `output` (string)
- `embedding_ids` (list of 3 ints: `[head_id, relation_id, tail_id]`)

Example:
```json
{
  "instruction": "Determine whether the triple is correct.",
  "input": "head, relation, tail",
  "output": "True",
  "embedding_ids": [12, 3, 98]
}
```

`embedding_ids` must be consistent with `kg/<dataset>/entity2id.txt` and `relation2id.txt`.

## KG format (OpenKE style)
Expected files under `kg/<dataset>/`:
- `entity2id.txt`
- `relation2id.txt`
- `train2id.txt` (default triple file used for adjacency)

By default, `train2id.txt` uses `head tail relation` ordering. Use `--triple_format` to change this (for example, `htr` or `hrt`).

If your relations are directional, use `--add_inverse` when building the KG index.
Without `--add_inverse`, the loader still adds both head and tail adjacency entries with the same relation id for neighborhood retrieval.

## Relation semantics
GMT uses relation semantic vectors for SGM:
- Provide a tensor file via `--rel_semantic_path` (shape `[num_relations, dim]`).
- If missing, it falls back to random vectors.
- Alternatively, use `--kge_model` to load relation embeddings from a KGE checkpoint.

Build relation semantics from relation definitions (optional):
```bash
python build_rel_semantics.py \
  --relation2id kg/CoDeX-S/relation2id.txt \
  --definitions data/CoDeX-S_rel_defs.json \
  --output_path data/CoDeX-S_rel_semantic.pt
```
This requires `sentence-transformers` and will download a model if not cached.

## Stage 1: SGM pretraining
Example (CoDeX-S):
```bash
python pretrain_gmt.py \
  --kg_dir kg/CoDeX-S \
  --output_dir checkpoints/sgm_codex \
  --rel_semantic_path data/CoDeX-S_rel_semantic.pt \
  --graph_dim 256 \
  --top_k 8 \
  --num_neg 64
```

Outputs:
- `checkpoints/sgm_codex/sgm_state.pth`
- `checkpoints/sgm_codex/sgm_config.json`

## Stage 2: Memory-augmented LLM finetuning
Example (CoDeX-S):
```bash
python finetune_gmt.py \
  --base_model "YOUR_LLM_PATH" \
  --data_path "data/CoDeX-S-train.json" \
  --output_dir checkpoints/gmt_codex \
  --kg_dir kg/CoDeX-S \
  --sgm_ckpt checkpoints/sgm_codex/sgm_state.pth \
  --memory_layers "24,25,26,27,28,29,30,31" \
  --lora_r 64 \
  --lora_alpha 128 \
  --learning_rate 3e-4 \
  --batch_size 12 \
  --micro_batch_size 12
```

Notes:
- `--memory_layers` is 0-based by default. Use `--memory_layers_1based` for 1-based indices.
- If `--memory_layers` is omitted, the top 8 layers are used.
- `--mask_query` removes the query triple from its local neighborhood.
- `--train_graph_module` allows SGM finetuning during Stage 2 (disabled by default).
- `--train_cross_attn_base` enables training the base cross-attention weights in addition to LoRA.
- Effective global batch follows:
  `batch_size = micro_batch_size * gradient_accumulation_steps * WORLD_SIZE`.
- For current script checks, `batch_size` must be divisible by `micro_batch_size`, and `(batch_size / micro_batch_size)` must be divisible by `WORLD_SIZE` in DDP.

Outputs:
- `checkpoints/gmt_codex/graph_encoder.pth`
- `checkpoints/gmt_codex/memory_attn.pth`
- `checkpoints/gmt_codex/gmt_config.json`

## Inference
Classification:
```bash
python inference_gmt.py \
  --base_model "YOUR_LLM_PATH" \
  --gmt_dir checkpoints/gmt_codex \
  --data_path "data/CoDeX-S-test.json" \
  --kg_dir kg/CoDeX-S \
  --max_new_tokens 16
```

Link prediction:
```bash
python inference_gmt.py \
  --base_model "YOUR_LLM_PATH" \
  --gmt_dir checkpoints/gmt_codex \
  --data_path "data/CoDeX-S-test.json" \
  --kg_dir kg/CoDeX-S \
  --task link_prediction \
  --hits_k "1,3,10"
```

Additional inference options:
- `--task auto` lets the script infer the task type.
- `--max_eval` limits the number of evaluated samples.
- `--mask_query` uses the same masking behavior as training.

## Checkpoints and reuse
Finetuning saves:
- `graph_encoder.pth` (SGM + memory tokenizer + projection)
- `memory_attn.pth` (memory cross-attention and gates)
- `gmt_config.json` (all run arguments)

Inference loads these files and reconstructs the GMT model on top of the base LLM.
Checkpoint loading now validates expected key sets for memory modules to avoid silent partial loads.

## Tips
- Ensure `embedding_ids` align with the KG ID mappings used in `kg/<dataset>/`.
- For new datasets, verify triple ordering and pass `--triple_format` if needed.
- If results are unstable, start by fixing `--seed` in `pretrain_gmt.py` and reducing `--top_k`.

## License
See `LICENSE`.
