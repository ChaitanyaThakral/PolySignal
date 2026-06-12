# -*- coding: utf-8 -*-
"""
graph/hetero_graph.py
=====================
Day 12 target: Build the PyG HeteroData graph from FAERS co-occurrence data.

Stubs only -- import-safe placeholder so the package is valid from Day 1.
"""
from __future__ import annotations


def build_hetero_graph(drug_features, event_features, edge_index):
    """
    Construct a torch_geometric.data.HeteroData object with:
      - 'drug' node type (features: drug embedding or one-hot)
      - 'event' node type (features: MedDRA PT embedding)
      - ('drug', 'reported_with', 'event') edge type

    Stub -- full implementation on Day 12.
    """
    raise NotImplementedError("Implement on Day 12")
