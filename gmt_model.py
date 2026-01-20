import json
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def _make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    batch_size, target_length = input_shape
    mask = torch.full(
        (target_length, target_length), float("-inf"), device=device
    )
    mask = torch.triu(mask, diagonal=1)
    if past_key_values_length > 0:
        past_mask = torch.zeros(
            target_length, past_key_values_length, device=device
        )
        mask = torch.cat([past_mask, mask], dim=-1)
    return mask[None, None, :, :].to(dtype).expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )


def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
) -> torch.Tensor:
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(
        batch_size, 1, tgt_len, src_len
    )
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        bias: bool = True,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = (
            nn.Parameter(torch.zeros(out_features)) if bias else None
        )
        self.weight.requires_grad = train_base
        if self.bias is not None:
            self.bias.requires_grad = train_base
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout)
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.lora_B(
                self.lora_A(self.lora_dropout(x))
            )
            result = result + lora_out * self.scaling
        return result


class MemoryCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        attn_dropout: float,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = LoRALinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            train_base=train_base,
        )
        self.k_proj = LoRALinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            train_base=train_base,
        )
        self.v_proj = LoRALinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            train_base=train_base,
        )
        self.o_proj = LoRALinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            train_base=train_base,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self, hidden_states: torch.Tensor, memory_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = hidden_states.size()
        _, mem_len, _ = memory_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(memory_states)
        value = self.v_proj(memory_states)
        query = query.view(
            batch_size, tgt_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, mem_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, mem_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_weights = torch.matmul(
            query, key.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.hidden_size)
        )
        return self.o_proj(attn_output)


class MemoryAugmentedLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config,
        use_memory: bool,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        train_cross_attn_base: bool,
    ) -> None:
        super().__init__()
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.use_memory = use_memory
        if use_memory:
            self.memory_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.memory_attn = MemoryCrossAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                attn_dropout=config.attention_dropout,
                train_base=train_cross_attn_base,
            )
            self.memory_gate = nn.Parameter(torch.zeros(1))
            self._init_memory_from_self_attn(base_layer)
        else:
            self.memory_layernorm = None
            self.memory_attn = None
            self.register_parameter("memory_gate", None)

    def _init_memory_from_self_attn(self, base_layer: nn.Module) -> None:
        if not hasattr(base_layer, "self_attn"):
            return
        attn = base_layer.self_attn
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return
        self.memory_attn.q_proj.weight.data.copy_(
            attn.q_proj.weight.data.to(
                device=self.memory_attn.q_proj.weight.device,
                dtype=self.memory_attn.q_proj.weight.dtype,
            )
        )
        self.memory_attn.k_proj.weight.data.copy_(
            attn.k_proj.weight.data.to(
                device=self.memory_attn.k_proj.weight.device,
                dtype=self.memory_attn.k_proj.weight.dtype,
            )
        )
        self.memory_attn.v_proj.weight.data.copy_(
            attn.v_proj.weight.data.to(
                device=self.memory_attn.v_proj.weight.device,
                dtype=self.memory_attn.v_proj.weight.dtype,
            )
        )
        self.memory_attn.o_proj.weight.data.copy_(
            attn.o_proj.weight.data.to(
                device=self.memory_attn.o_proj.weight.device,
                dtype=self.memory_attn.o_proj.weight.dtype,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        graph_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output
        if self.use_memory and graph_memory is not None:
            memory_input = self.memory_layernorm(hidden_states)
            memory_output = self.memory_attn(memory_input, graph_memory)
            hidden_states = hidden_states + self.memory_gate * memory_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        return outputs


class LlamaWithMemoryModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        memory_layers: List[int],
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        train_cross_attn_base: bool,
    ) -> None:
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.embed_tokens
        self.layers = nn.ModuleList([])
        memory_layer_set = set(memory_layers)
        for idx, layer in enumerate(base_model.layers):
            self.layers.append(
                MemoryAugmentedLlamaDecoderLayer(
                    base_layer=layer,
                    config=self.config,
                    use_memory=idx in memory_layer_set,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    train_cross_attn_base=train_cross_attn_base,
                )
            )
        self.norm = base_model.norm

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attention_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attention_mask
                if combined_attention_mask is None
                else expanded_attention_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph_memory: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values else 0
        )
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                device=inputs_embeds.device
                if inputs_embeds is not None
                else input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(
                batch_size, -1
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        hidden_states = inputs_embeds
        graph_memory_cache = {} if graph_memory is not None else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_graph_memory = None
            if graph_memory_cache is not None and decoder_layer.memory_attn is not None:
                target_device = hidden_states.device
                layer_graph_memory = graph_memory_cache.get(target_device)
                if layer_graph_memory is None:
                    layer_graph_memory = graph_memory.to(target_device)
                    graph_memory_cache[target_device] = layer_graph_memory
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx]
                if past_key_values
                else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                graph_memory=layer_graph_memory,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            outputs = (hidden_states, next_decoder_cache)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            return outputs
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class RelationContextLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, states: torch.Tensor, contexts: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_edges, dim = states.size()
        query = states.view(batch_size * num_edges, 1, dim)
        key = contexts.view(batch_size * num_edges, 1, dim)
        value = contexts.view(batch_size * num_edges, 1, dim)
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        attn_out = attn_out.view(batch_size, num_edges, dim)
        states = states + self.dropout(attn_out)
        states = self.norm1(states)
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(states))))
        states = states + self.dropout(ff_out)
        states = self.norm2(states)
        if mask is not None:
            states = states * mask.unsqueeze(-1)
        return states


class SemanticGraphModule(nn.Module):
    def __init__(
        self,
        kg_index,
        rel_semantics: torch.Tensor,
        graph_dim: int,
        num_layers: int,
        num_heads: int,
        top_k: int,
        dropout: float,
        trainable_rel_semantics: bool = False,
    ) -> None:
        super().__init__()
        self.kg_index = kg_index
        self.top_k = top_k
        self.graph_dim = graph_dim
        rel_dim = rel_semantics.size(-1)
        self.rel_semantic = nn.Embedding(
            rel_semantics.size(0), rel_dim
        )
        self.rel_semantic.weight.data.copy_(rel_semantics)
        self.rel_semantic.weight.requires_grad = trainable_rel_semantics
        self.rel_proj = (
            nn.Linear(rel_dim, graph_dim) if rel_dim != graph_dim else None
        )
        self.layers = nn.ModuleList(
            [
                RelationContextLayer(graph_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def _rel_vector(self, rel_ids: torch.Tensor) -> torch.Tensor:
        rel_vec = self.rel_semantic(rel_ids)
        if self.rel_proj is not None:
            rel_vec = self.rel_proj(rel_vec)
        return rel_vec

    def _encode_edges(
        self, edges: List[Tuple[int, int]], device: torch.device
    ) -> torch.Tensor:
        rel_vecs = []
        neighbor_aggs = []
        for rel_id, neighbor_id in edges:
            rel_vec = self._rel_vector(
                torch.tensor(rel_id, device=device)
            )
            neighbor_edges = self.kg_index.adj[neighbor_id]
            if neighbor_edges:
                neighbor_rel_ids = torch.tensor(
                    [edge[0] for edge in neighbor_edges],
                    dtype=torch.long,
                    device=device,
                )
                neighbor_vecs = self._rel_vector(neighbor_rel_ids)
                rel_norm = F.normalize(rel_vec, dim=-1)
                neigh_norm = F.normalize(neighbor_vecs, dim=-1)
                sims = torch.matmul(neigh_norm, rel_norm)
                top_k = min(self.top_k, sims.numel())
                if top_k > 0:
                    top_idx = torch.topk(sims, k=top_k).indices
                    neighbor_agg = neighbor_vecs.index_select(0, top_idx).mean(dim=0)
                else:
                    neighbor_agg = torch.zeros(self.graph_dim, device=device)
            else:
                neighbor_agg = torch.zeros(self.graph_dim, device=device)
            rel_vecs.append(rel_vec)
            neighbor_aggs.append(neighbor_agg)
        if not rel_vecs:
            return torch.zeros(0, self.graph_dim, device=device)
        rel_vecs = torch.stack(rel_vecs, dim=0)
        neighbor_aggs = torch.stack(neighbor_aggs, dim=0)
        edge_states = rel_vecs.unsqueeze(0)
        neighbor_ctx = neighbor_aggs.unsqueeze(0)
        mask = torch.ones(edge_states.size()[:-1], device=device)
        for layer in self.layers:
            edge_states = layer(edge_states, neighbor_ctx, mask=mask)
        return edge_states.squeeze(0)

    def _collect_edges(
        self, head_id: int, rel_id: int, tail_id: int, mask_query: bool
    ) -> List[Tuple[int, int]]:
        edges = []
        edges.extend(self.kg_index.adj[head_id])
        edges.extend(self.kg_index.adj[tail_id])
        if not mask_query:
            return edges
        filtered = []
        if self.kg_index.add_inverse:
            base_rel = rel_id % self.kg_index.base_num_relations
            inv_rel = base_rel + self.kg_index.base_num_relations
            rel_set = {base_rel, inv_rel}
        else:
            rel_set = {rel_id}
        for r_id, n_id in edges:
            if n_id in (head_id, tail_id) and r_id in rel_set:
                continue
            filtered.append((r_id, n_id))
        return filtered

    def forward(
        self, triple_ids: torch.Tensor, mask_query: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.rel_semantic.weight.device
        batch_states = []
        lengths = []
        for triple in triple_ids.tolist():
            head_id, rel_id, tail_id = triple
            edges = self._collect_edges(
                head_id, rel_id, tail_id, mask_query=mask_query
            )
            edge_states = self._encode_edges(edges, device=device)
            batch_states.append(edge_states)
            lengths.append(edge_states.size(0))
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            empty = torch.zeros(
                triple_ids.size(0), 1, self.graph_dim, device=device
            )
            mask = torch.zeros(triple_ids.size(0), 1, device=device)
            return empty, mask
        padded = torch.zeros(
            triple_ids.size(0), max_len, self.graph_dim, device=device
        )
        mask = torch.zeros(triple_ids.size(0), max_len, device=device)
        for idx, edge_states in enumerate(batch_states):
            if edge_states.numel() == 0:
                continue
            length = edge_states.size(0)
            padded[idx, :length] = edge_states
            mask[idx, :length] = 1.0
        return padded, mask

    def encode_entity(self, entity_ids: torch.Tensor) -> torch.Tensor:
        device = self.rel_semantic.weight.device
        outputs = []
        for ent_id in entity_ids.tolist():
            edges = self.kg_index.adj[ent_id]
            edge_states = self._encode_edges(edges, device=device)
            if edge_states.numel() == 0:
                outputs.append(torch.zeros(self.graph_dim, device=device))
            else:
                outputs.append(edge_states.mean(dim=0))
        return torch.stack(outputs, dim=0)

    def relation_semantics(self, rel_ids: torch.Tensor) -> torch.Tensor:
        return self._rel_vector(rel_ids)


class GraphMemoryTokenizer(nn.Module):
    def __init__(self, num_memory_tokens: int, graph_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.memory_queries = nn.Parameter(
            torch.randn(num_memory_tokens, graph_dim) * 0.02
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=graph_dim, num_heads=num_heads, batch_first=True
        )

    def forward(
        self, edge_states: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = edge_states.size(0)
        if edge_states.size(1) == 0 or edge_mask.sum() == 0:
            return edge_states.new_zeros(
                batch_size, self.num_memory_tokens, edge_states.size(-1)
            )
        query = self.memory_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        key_padding_mask = edge_mask == 0
        memory, _ = self.attn(
            query, edge_states, edge_states, key_padding_mask=key_padding_mask
        )
        no_edges = edge_mask.sum(dim=1) == 0
        if no_edges.any():
            memory = memory.masked_fill(no_edges[:, None, None], 0.0)
        return memory


class GraphMemoryEncoder(nn.Module):
    def __init__(
        self,
        kg_index,
        rel_semantics: torch.Tensor,
        graph_dim: int,
        num_layers: int,
        num_heads: int,
        top_k: int,
        num_memory_tokens: int,
        memory_heads: int,
        llm_hidden_size: int,
        dropout: float,
        trainable_rel_semantics: bool,
    ) -> None:
        super().__init__()
        self.sgm = SemanticGraphModule(
            kg_index=kg_index,
            rel_semantics=rel_semantics,
            graph_dim=graph_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            top_k=top_k,
            dropout=dropout,
            trainable_rel_semantics=trainable_rel_semantics,
        )
        self.tokenizer = GraphMemoryTokenizer(
            num_memory_tokens=num_memory_tokens,
            graph_dim=graph_dim,
            num_heads=memory_heads,
        )
        self.proj = nn.Linear(graph_dim, llm_hidden_size)

    def forward(self, triple_ids: torch.Tensor, mask_query: bool) -> torch.Tensor:
        edge_states, edge_mask = self.sgm(triple_ids, mask_query=mask_query)
        memory = self.tokenizer(edge_states, edge_mask)
        return self.proj(memory)


class GMTForCausalLM(nn.Module):
    def __init__(
        self,
        base_model,
        graph_encoder: GraphMemoryEncoder,
        memory_layers: List[int],
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        train_cross_attn_base: bool,
    ) -> None:
        super().__init__()
        self.config = base_model.config
        self.graph_encoder = graph_encoder
        self.default_mask_query = True
        self.model = LlamaWithMemoryModel(
            base_model=base_model.model,
            memory_layers=memory_layers,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            train_cross_attn_base=train_cross_attn_base,
        )
        self.lm_head = base_model.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: Optional[torch.LongTensor] = None,
        graph_memory: Optional[torch.Tensor] = None,
        mask_query: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        if mask_query is None:
            mask_query = self.default_mask_query
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )
        if graph_memory is None:
            if embedding_ids is None:
                raise ValueError("embedding_ids is required when graph_memory is None")
            graph_memory = self.graph_encoder(embedding_ids, mask_query=mask_query)
        graph_memory = graph_memory.to(self.lm_head.weight.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_memory=graph_memory,
        )
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (loss, logits) + outputs[1:]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_memory_state_dict(self) -> dict:
        state = {}
        for name, tensor in self.state_dict().items():
            if "memory_attn" in name or "memory_gate" in name or "memory_layernorm" in name:
                state[name] = tensor
        return state

    def align_memory_devices(self) -> None:
        for layer in self.model.layers:
            if layer.memory_attn is None:
                continue
            target_device = next(layer.self_attn.parameters()).device
            target_dtype = next(layer.self_attn.parameters()).dtype
            layer.memory_attn.to(device=target_device, dtype=target_dtype)
            layer.memory_layernorm.to(device=target_device, dtype=target_dtype)
            layer.memory_gate = nn.Parameter(
                layer.memory_gate.to(device=target_device, dtype=target_dtype)
            )

    def save_gmt(self, output_dir: str, config: dict) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.graph_encoder.state_dict(),
            os.path.join(output_dir, "graph_encoder.pth"),
        )
        torch.save(
            self.get_memory_state_dict(),
            os.path.join(output_dir, "memory_attn.pth"),
        )
        with open(os.path.join(output_dir, "gmt_config.json"), "w") as f:
            json.dump(config, f, indent=2)
