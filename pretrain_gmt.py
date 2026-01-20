import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .gmt_model import SemanticGraphModule
    from .graph_data import load_kg_index, load_relation_semantics, load_triples
except ImportError:  # Allows running as a script.
    from gmt_model import SemanticGraphModule
    from graph_data import load_kg_index, load_relation_semantics, load_triples


class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return torch.tensor(self.triples[idx], dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SGM for GMT")
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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable_rel_semantics", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_neg", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def sample_negatives(
    heads, rels, tails, num_entities, num_neg, device
):
    batch_size = heads.size(0)
    neg_heads = heads.unsqueeze(1).repeat(1, num_neg)
    neg_tails = tails.unsqueeze(1).repeat(1, num_neg)
    mask = torch.rand(batch_size, num_neg, device=device) < 0.5
    if mask.any():
        neg_heads[mask] = torch.randint(
            0, num_entities, (int(mask.sum()),), device=device
        )
    inv_mask = ~mask
    if inv_mask.any():
        neg_tails[inv_mask] = torch.randint(
            0, num_entities, (int(inv_mask.sum()),), device=device
        )
    neg_rels = rels.unsqueeze(1).repeat(1, num_neg)
    return neg_heads.view(-1), neg_rels.view(-1), neg_tails.view(-1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
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
    sgm = SemanticGraphModule(
        kg_index=kg_index,
        rel_semantics=rel_semantics,
        graph_dim=args.graph_dim,
        num_layers=args.sgm_layers,
        num_heads=args.sgm_heads,
        top_k=args.top_k,
        dropout=args.dropout,
        trainable_rel_semantics=args.trainable_rel_semantics,
    ).to(device)
    triple_path = os.path.join(args.kg_dir, args.triple_file)
    triples = load_triples(triple_path, triple_format=args.triple_format)
    loader = DataLoader(
        TripleDataset(triples),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(sgm.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        sgm.train()
        total_loss = 0.0
        step_count = 0
        for batch in loader:
            batch = batch.to(device)
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            e_h = sgm.encode_entity(heads)
            e_t = sgm.encode_entity(tails)
            e_r = sgm.relation_semantics(rels)
            pos_score = (e_h * e_r * e_t).sum(dim=-1)
            neg_heads, neg_rels, neg_tails = sample_negatives(
                heads,
                rels,
                tails,
                kg_index.num_entities,
                args.num_neg,
                device,
            )
            e_hn = sgm.encode_entity(neg_heads)
            e_tn = sgm.encode_entity(neg_tails)
            e_rn = sgm.relation_semantics(neg_rels)
            neg_score = (e_hn * e_rn * e_tn).sum(dim=-1)
            neg_score = neg_score.view(heads.size(0), args.num_neg)
            loss = -(
                F.logsigmoid(pos_score)
                + F.logsigmoid(-neg_score).mean(dim=1)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
        avg_loss = total_loss / max(step_count, 1)
        print(f"Epoch {epoch + 1} loss {avg_loss:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {"sgm": sgm.state_dict()},
        os.path.join(args.output_dir, "sgm_state.pth"),
    )
    config = vars(args)
    with open(os.path.join(args.output_dir, "sgm_config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
