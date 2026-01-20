import json
import os
from typing import List, Tuple

import torch


class KGIndex:
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        triples: List[Tuple[int, int, int]],
        add_inverse: bool = False,
    ) -> None:
        self.num_entities = num_entities
        self.base_num_relations = num_relations
        self.add_inverse = add_inverse
        self.num_relations = (
            num_relations * 2 if add_inverse else num_relations
        )
        self.adj = [[] for _ in range(num_entities)]
        for head_id, rel_id, tail_id in triples:
            self.adj[head_id].append((rel_id, tail_id))
            if add_inverse:
                inv_rel = rel_id + num_relations
                self.adj[tail_id].append((inv_rel, head_id))
            else:
                self.adj[tail_id].append((rel_id, head_id))


def load_id_map(path: str) -> dict:
    mapping = {}
    with open(path, "r") as f:
        first = f.readline()
        if not first:
            return mapping
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name, idx = parts[0], int(parts[1])
            mapping[name] = idx
    return mapping


def load_count_from_map(path: str) -> int:
    with open(path, "r") as f:
        first = f.readline()
    return int(first.strip())


def load_triples(path: str, triple_format: str = "htr") -> List[Tuple[int, int, int]]:
    if len(triple_format) != 3 or set(triple_format) != {"h", "r", "t"}:
        raise ValueError("triple_format must be a permutation of 'h', 'r', 't'")
    triples = []
    with open(path, "r") as f:
        first = f.readline()
        if not first:
            return triples
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            values = list(map(int, parts))
            mapping = dict(zip(triple_format, values))
            triples.append((mapping["h"], mapping["r"], mapping["t"]))
    return triples


def load_kg_index(
    kg_dir: str,
    triple_file: str = "train2id.txt",
    triple_format: str = "htr",
    add_inverse: bool = False,
) -> KGIndex:
    entity_path = os.path.join(kg_dir, "entity2id.txt")
    relation_path = os.path.join(kg_dir, "relation2id.txt")
    triple_path = os.path.join(kg_dir, triple_file)
    num_entities = load_count_from_map(entity_path)
    num_relations = load_count_from_map(relation_path)
    triples = load_triples(triple_path, triple_format=triple_format)
    return KGIndex(
        num_entities=num_entities,
        num_relations=num_relations,
        triples=triples,
        add_inverse=add_inverse,
    )


def load_relation_semantics(
    num_relations: int,
    rel_semantic_path: str = "",
    rel_semantic_dim: int = 256,
    add_inverse: bool = False,
    kge_model: str = "",
) -> torch.Tensor:
    rel_emb = None
    if rel_semantic_path and os.path.exists(rel_semantic_path):
        data = torch.load(rel_semantic_path, map_location="cpu")
        if isinstance(data, dict) and "rel_embeddings" in data:
            rel_emb = data["rel_embeddings"]
        else:
            rel_emb = data
    elif kge_model:
        try:
            from .process_kge import load_pretrain_kge
        except ImportError:  # Allows running as a script.
            from process_kge import load_pretrain_kge

        _, rel_emb = load_pretrain_kge(kge_model)
    if rel_emb is None:
        rel_emb = torch.randn(num_relations, rel_semantic_dim) * 0.02
    if add_inverse:
        rel_emb = torch.cat([rel_emb, rel_emb.clone()], dim=0)
    return rel_emb


def load_relation_definitions(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
