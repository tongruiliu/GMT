import argparse
import json
import os
import re

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

try:
    from .gmt_model import GMTForCausalLM, GraphMemoryEncoder
    from .graph_data import load_kg_index, load_relation_semantics
    from .utils.prompter import Prompter
except ImportError:  # Allows running as a script.
    from gmt_model import GMTForCausalLM, GraphMemoryEncoder
    from graph_data import load_kg_index, load_relation_semantics
    from utils.prompter import Prompter


def parse_args():
    parser = argparse.ArgumentParser(description="GMT inference")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--gmt_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--kg_dir", type=str, required=True)
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
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_layers", type=str, default="")
    parser.add_argument("--memory_layers_1based", action="store_true")
    parser.add_argument("--mask_query", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--prompt_template_name", type=str, default="alpaca")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "link_prediction", "auto"],
    )
    parser.add_argument("--hits_k", type=str, default="1,3,10")
    parser.add_argument("--max_eval", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_layers(value: str, one_based: bool):
    if not value:
        return []
    layers = [int(v.strip()) for v in value.split(",") if v.strip()]
    if one_based:
        layers = [v - 1 for v in layers]
    return layers


def validate_memory_layers(layers: list, num_layers: int) -> list:
    if len(set(layers)) != len(layers):
        raise ValueError(f"memory_layers contains duplicates: {layers}")
    invalid = [v for v in layers if v < 0 or v >= num_layers]
    if invalid:
        raise ValueError(
            f"memory_layers out of range for model with {num_layers} layers: {invalid}"
        )
    return layers


def load_config(gmt_dir: str):
    config_path = os.path.join(gmt_dir, "gmt_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def load_dataset(path: str):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    with open(path, "r") as f:
        return json.load(f)


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids,
    attention_mask,
    embedding_ids,
    max_new_tokens,
):
    graph_memory = model.graph_encoder(
        embedding_ids, mask_query=model.default_mask_query
    )
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_memory=graph_memory,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )
    return input_ids


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s\-\*\d\.\)\:]+", "", text)
    text = text.strip().strip('"').strip("'")
    return text.lower()


def split_ranked_list(text: str) -> list:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    if not lines:
        lines = [text]
    raw_items = []
    for line in lines:
        parts = re.split(r"[;,]\s*", line)
        raw_items.extend([p for p in parts if p.strip()])
    cleaned = [normalize_text(item) for item in raw_items if item.strip()]
    seen = set()
    output = []
    for item in cleaned:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def parse_candidates(data: dict) -> list:
    for key in ("candidates", "candidate_entities", "candidate_list"):
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]
    input_text = data.get("input", "")
    match = re.search(r"\{([^}]+)\}", input_text)
    if match:
        items = match.group(1).split(",")
        return [item.strip() for item in items if item.strip()]
    return []


def parse_gold_answers(data: dict) -> list:
    for key in ("answers", "answer", "label"):
        if key in data:
            value = data[key]
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]
    output = data.get("output", "")
    if isinstance(output, list):
        return [str(v) for v in output]
    output = str(output)
    if output.strip().startswith("[") and output.strip().endswith("]"):
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
    return [item for item in split_ranked_list(output) if item]


def map_predictions_to_candidates(pred_items: list, candidates: list) -> list:
    if not candidates:
        return pred_items
    normalized_candidates = {normalize_text(c): c for c in candidates}
    mapped = []
    for item in pred_items:
        if item in normalized_candidates:
            mapped.append(normalized_candidates[item])
            continue
        for cand_norm, cand in normalized_candidates.items():
            if cand_norm and cand_norm in item:
                mapped.append(cand)
                break
    seen = set()
    output = []
    for item in mapped:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def detect_task(data: dict) -> str:
    input_text = data.get("input", "").lower()
    if "candidate" in input_text or "candidates" in input_text:
        return "link_prediction"
    output = str(data.get("output", "")).lower()
    if "true" in output or "false" in output:
        return "classification"
    return "link_prediction"


def compute_classification_metrics(labels, preds):
    tp = sum(1 for y, yhat in zip(labels, preds) if y == 1 and yhat == 1)
    tn = sum(1 for y, yhat in zip(labels, preds) if y == 0 and yhat == 0)
    fp = sum(1 for y, yhat in zip(labels, preds) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(labels, preds) if y == 1 and yhat == 0)
    total = max(len(labels), 1)
    acc = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return acc, precision, recall, f1


def compute_link_metrics(results, hits_k):
    hits_counts = {k: 0 for k in hits_k}
    mrr_total = 0.0
    valid = 0
    for item in results:
        preds = item["preds"]
        golds = item["golds"]
        if not preds or not golds:
            continue
        norm_preds = [normalize_text(p) for p in preds]
        norm_golds = [normalize_text(g) for g in golds]
        rank = None
        for idx, pred in enumerate(norm_preds, start=1):
            if pred in norm_golds:
                rank = idx
                break
        if rank is None:
            rank = float("inf")
        for k in hits_k:
            if rank <= k:
                hits_counts[k] += 1
        if rank != float("inf"):
            mrr_total += 1.0 / rank
        valid += 1
    if valid == 0:
        return None
    hits = {k: hits_counts[k] / valid for k in hits_k}
    mrr = mrr_total / valid
    return hits, mrr, valid


def main():
    args = parse_args()
    config = load_config(args.gmt_dir)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    memory_layers_arg = args.memory_layers or config.get("memory_layers", "")
    memory_layers_1based = args.memory_layers_1based or config.get(
        "memory_layers_1based", False
    )
    memory_layers = parse_layers(memory_layers_arg, one_based=memory_layers_1based)
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
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to(device)
    if memory_layers:
        memory_layers = validate_memory_layers(
            memory_layers, base_model.config.num_hidden_layers
        )
        effective_memory_layers = memory_layers
    else:
        effective_memory_layers = list(
            range(
                max(base_model.config.num_hidden_layers - 8, 0),
                base_model.config.num_hidden_layers,
            )
        )
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
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
        trainable_rel_semantics=False,
    )
    model = GMTForCausalLM(
        base_model=base_model,
        graph_encoder=graph_encoder,
        memory_layers=effective_memory_layers,
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_dropout=config.get("lora_dropout", 0.01),
        train_cross_attn_base=False,
    ).to(device)
    model.default_mask_query = args.mask_query
    model.align_memory_devices()
    graph_state = torch.load(
        os.path.join(args.gmt_dir, "graph_encoder.pth"),
        map_location="cpu",
    )
    model.graph_encoder.load_state_dict(graph_state, strict=True)
    memory_state = torch.load(
        os.path.join(args.gmt_dir, "memory_attn.pth"),
        map_location="cpu",
    )
    expected_memory_keys = set(model.get_memory_state_dict().keys())
    loaded_memory_keys = set(memory_state.keys())
    missing_memory_keys = sorted(expected_memory_keys - loaded_memory_keys)
    unexpected_memory_keys = sorted(loaded_memory_keys - expected_memory_keys)
    if missing_memory_keys or unexpected_memory_keys:
        raise RuntimeError(
            "Memory checkpoint keys mismatch. "
            f"missing={missing_memory_keys[:10]} "
            f"unexpected={unexpected_memory_keys[:10]}"
        )
    model.load_state_dict(memory_state, strict=False)
    model.align_memory_devices()
    model.eval()

    prompter = Prompter(args.prompt_template_name)
    dataset = load_dataset(args.data_path)
    results = []
    max_eval = args.max_eval if args.max_eval > 0 else None
    for data in dataset:
        if max_eval is not None and len(results) >= max_eval:
            break
        prompt = prompter.generate_prompt(
            data["instruction"], data["input"]
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        embedding_ids = torch.tensor(
            data["embedding_ids"], dtype=torch.long
        ).unsqueeze(0).to(device)
        prompt_len = input_ids.size(-1)
        gen_ids = greedy_generate(
            model,
            input_ids,
            attention_mask,
            embedding_ids,
            args.max_new_tokens,
        )
        response = tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()
        results.append({"data": data, "predict": response})
        print(response)
    task = args.task
    if task == "auto":
        task = detect_task(results[0]["data"]) if results else "classification"
    if task == "classification":
        true_labels = []
        pred_labels = []
        for item in results:
            answer = str(item["data"].get("output", ""))
            predict = item["predict"]
            true_labels.append(1 if "true" in answer.lower() else 0)
            pred_labels.append(1 if "true" in predict.lower() else 0)
        if true_labels:
            acc, precision, recall, f1 = compute_classification_metrics(
                true_labels, pred_labels
            )
            print(
                f"Accuracy: {acc:.4f} Precision: {precision:.4f} "
                f"Recall: {recall:.4f} F1: {f1:.4f}"
            )
    else:
        hits_k = [int(k) for k in args.hits_k.split(",") if k.strip()]
        metric_rows = []
        for item in results:
            data = item["data"]
            candidates = parse_candidates(data)
            golds = parse_gold_answers(data)
            pred_items = split_ranked_list(item["predict"])
            pred_items = map_predictions_to_candidates(pred_items, candidates)
            metric_rows.append({"preds": pred_items, "golds": golds})
        metrics = compute_link_metrics(metric_rows, hits_k)
        if metrics is not None:
            hits, mrr, valid = metrics
            hits_str = " ".join(
                [f"Hits@{k}:{hits[k]:.4f}" for k in hits_k]
            )
            print(f"{hits_str} MRR:{mrr:.4f} (n={valid})")


if __name__ == "__main__":
    main()
