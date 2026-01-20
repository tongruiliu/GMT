import argparse
import json
import os
from typing import List

import torch
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

try:
    from .gmt_model import GMTForCausalLM, GraphMemoryEncoder
    from .graph_data import load_kg_index, load_relation_semantics
    from .utils.prompter import Prompter
except ImportError:  # Allows running as a script.
    from gmt_model import GMTForCausalLM, GraphMemoryEncoder
    from graph_data import load_kg_index, load_relation_semantics
    from utils.prompter import Prompter


def parse_layers(value: str, one_based: bool) -> List[int]:
    if not value:
        return []
    layers = [int(v.strip()) for v in value.split(",") if v.strip()]
    if one_based:
        layers = [v - 1 for v in layers]
    return layers


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune GMT (Stage 2)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--kg_dir", type=str, required=True)
    parser.add_argument("--sgm_ckpt", type=str, default="")
    parser.add_argument("--triple_file", type=str, default="train2id.txt")
    parser.add_argument("--triple_format", type=str, default="htr")
    parser.add_argument("--add_inverse", action="store_true")
    parser.add_argument("--rel_semantic_path", type=str, default="")
    parser.add_argument("--rel_semantic_dim", type=int, default=256)
    parser.add_argument("--kge_model", type=str, default="")
    parser.add_argument("--graph_dim", type=int, default=256)
    parser.add_argument("--sgm_layers", type=int, default=2)
    parser.add_argument("--sgm_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--graph_dropout", type=float, default=0.1)
    parser.add_argument("--trainable_rel_semantics", action="store_true")
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_layers", type=str, default="")
    parser.add_argument("--memory_layers_1based", action="store_true")
    parser.add_argument("--mask_query", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    parser.add_argument("--train_cross_attn_base", action="store_true")
    parser.add_argument("--train_graph_module", action="store_true")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--micro_batch_size", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--val_set_size", type=int, default=0)
    parser.add_argument("--train_on_inputs", action="store_true", default=True)
    parser.add_argument("--no_train_on_inputs", action="store_false", dest="train_on_inputs")
    parser.add_argument("--add_eos_token", action="store_true")
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca")
    return parser.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(json.dumps(vars(args), indent=2))
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    prompter = Prompter(args.prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=args.add_eos_token)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if args.add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len
                + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        tokenized_full_prompt["embedding_ids"] = data_point["embedding_ids"]
        return tokenized_full_prompt

    kg_index = load_kg_index(
        kg_dir=args.kg_dir,
        triple_file=args.triple_file,
        triple_format=args.triple_format,
        add_inverse=args.add_inverse,
    )
    rel_semantics = load_relation_semantics(
        num_relations=kg_index.base_num_relations,
        rel_semantic_path=args.rel_semantic_path,
        rel_semantic_dim=args.rel_semantic_dim,
        add_inverse=args.add_inverse,
        kge_model=args.kge_model,
    )
    graph_encoder = GraphMemoryEncoder(
        kg_index=kg_index,
        rel_semantics=rel_semantics,
        graph_dim=args.graph_dim,
        num_layers=args.sgm_layers,
        num_heads=args.sgm_heads,
        top_k=args.top_k,
        num_memory_tokens=args.num_memory_tokens,
        memory_heads=args.memory_heads,
        llm_hidden_size=base_model.config.hidden_size,
        dropout=args.graph_dropout,
        trainable_rel_semantics=args.trainable_rel_semantics,
    )
    if args.sgm_ckpt:
        state = torch.load(args.sgm_ckpt, map_location="cpu")
        if isinstance(state, dict) and "sgm" in state:
            graph_encoder.sgm.load_state_dict(state["sgm"], strict=False)
        else:
            graph_encoder.sgm.load_state_dict(state, strict=False)
    num_layers = base_model.config.num_hidden_layers
    if args.memory_layers:
        memory_layers = parse_layers(
            args.memory_layers, one_based=args.memory_layers_1based
        )
        memory_layers_str = args.memory_layers
        memory_layers_1based = args.memory_layers_1based
    else:
        start = max(num_layers - 8, 0)
        memory_layers = list(range(start, num_layers))
        memory_layers_str = ",".join(str(v) for v in memory_layers)
        memory_layers_1based = False
    model = GMTForCausalLM(
        base_model=base_model,
        graph_encoder=graph_encoder,
        memory_layers=memory_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_cross_attn_base=args.train_cross_attn_base,
    )
    model.default_mask_query = args.mask_query
    graph_device = next(base_model.parameters()).device
    model.graph_encoder.to(graph_device)
    model.align_memory_devices()
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    for param in model.graph_encoder.tokenizer.parameters():
        param.requires_grad = True
    for param in model.graph_encoder.proj.parameters():
        param.requires_grad = True
    for param in model.graph_encoder.sgm.parameters():
        param.requires_grad = args.train_graph_module
    for name, param in model.named_parameters():
        if "memory_attn" in name:
            if "lora_" in name:
                param.requires_grad = True
            elif args.train_cross_attn_base:
                param.requires_grad = True
        elif "memory_gate" in name or "memory_layernorm" in name:
            param.requires_grad = True

    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)
    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    def data_collator(features):
        embedding_ids = torch.tensor(
            [f["embedding_ids"] for f in features], dtype=torch.long
        )
        batch = tokenizer.pad(
            [
                {k: v for k, v in f.items() if k != "embedding_ids"}
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )
        if "labels" in batch:
            batch["labels"][batch["attention_mask"] == 0] = -100
        batch["embedding_ids"] = embedding_ids
        return batch

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_hf",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="no",
            output_dir=args.output_dir,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to=None,
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    save_config = vars(args)
    save_config["memory_layers"] = memory_layers_str
    save_config["memory_layers_1based"] = memory_layers_1based
    model.save_gmt(args.output_dir, save_config)


if __name__ == "__main__":
    main()
