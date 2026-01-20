import argparse
import json
import os

import torch

try:
    from .graph_data import load_id_map
except ImportError:  # Allows running as a script.
    from graph_data import load_id_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build relation semantic embeddings from definitions"
    )
    parser.add_argument("--relation2id", type=str, required=True)
    parser.add_argument("--definitions", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_definitions(path: str) -> dict:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    definitions = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                rel, desc = line.split("\t", 1)
            else:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    rel, desc = parts
                else:
                    continue
            definitions[rel] = desc
    return definitions


def main():
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc

    rel_map = load_id_map(args.relation2id)
    definitions = load_definitions(args.definitions)
    rel_names = [None] * len(rel_map)
    for name, idx in rel_map.items():
        rel_names[idx] = name
    texts = []
    for name in rel_names:
        if name is None:
            desc = ""
        else:
            desc = definitions.get(name, name)
        texts.append(desc)
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True
    )
    rel_embeddings = torch.tensor(embeddings)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"rel_embeddings": rel_embeddings}, args.output_path)
    print(f"Saved {rel_embeddings.shape} to {args.output_path}")


if __name__ == "__main__":
    main()
