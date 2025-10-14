from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dgl
import numpy as np


class KAGnnGapPredictor:
    """Minimal wrapper to load the KA-GNN regressor and predict ST Gap for SMILES."""

    _MODEL_MAPPING = {
        "ka_gnn": "KA_GNN",
        "ka_gnn_two": "KA_GNN_two",
        "mlp_sage": "MLPGNN",
        "mlp_sage_two": "MLPGNN_two",
        "kan_sage": "KANGNN",
        "kan_sage_two": "KANGNN_two",
    }

    _DATASET_DIM = {
        "tox21": 12,
        "muv": 17,
        "sider": 27,
        "clintox": 2,
        "bace": 1,
        "bbbp": 1,
        "hiv": 1,
        "dft": 2,
    }

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        cfg = dict(config or {})
        # Always keep raw invalid predictions as NaN so downstream normalization can
        # distinguish them from genuine zero-valued gaps (which should map to score=1).
        self._invalid_value = float("nan")

        # Resolve KA-GNN project root. Default to repository directory that hosts the
        # predictor ("prediction_model"), but keep supporting custom overrides and the
        # legacy expectation where assets live under a nested "KA-GNN" folder.
        default_root = Path(__file__).resolve().parent
        root_override = cfg.get("root")
        if root_override is None:
            legacy_root = default_root
            root = legacy_root if legacy_root.exists() else default_root
        else:
            root = Path(root_override)
            if not root.is_absolute():
                root = (default_root / root).resolve()
        self._root = root
        """
        model_path = cfg.get("model_path")
        if model_path is None:
            model_path = self._root / "model.pth"
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = (self._root / model_path).resolve()
        """
        model_path = "./data_preparation/prediction_model/pretrained_models/best.pth"
        self._model_path = model_path

        self._encoder_atom = cfg.get("encoder_atom", "cgcnn")
        self._encoder_bond = cfg.get("encoder_bond", "dim_14")
        self._force_field = cfg.get("force_field", "mmff")

        self._device = None
        self._model = None
        self._gap_index = None
        self._path_complex_mol = None
        self._update_node_features = None

        self._load_assets()

    def _load_assets(self) -> None:
        import torch

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure KA-GNN modules are importable.
        root_str = str(self._root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            graph_module = importlib.import_module("utils_kan.graph_path")
        except ModuleNotFoundError:
            graph_module = importlib.import_module("utils.graph_path")
        model_module = importlib.import_module("model.ka_gnn")

        if not hasattr(graph_module, "path_complex_mol"):
            raise AttributeError("utils.graph_path must expose path_complex_mol")
        self._path_complex_mol = graph_module.path_complex_mol

        self._update_node_features = self._make_update_node_features()

        checkpoint = torch.load(str(self._model_path), map_location="cpu")
        args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

        model_select = args.get("model_select", "ka_gnn")
        model_cls_name = self._MODEL_MAPPING.get(model_select, "KA_GNN")
        model_cls = getattr(model_module, model_cls_name)

        dataset = args.get("select_dataset", "dft")
        label_dim = int(args.get("label_dim", self._DATASET_DIM.get(dataset, 1)))
        is_dft = dataset == "dft"
        target_dim = label_dim + (1 if is_dft else 0)
        self._gap_index = label_dim if is_dft else max(0, target_dim - 1)

        grid_feat = int(args.get("grid_feat", 1))
        num_layers = int(args.get("num_layers", 4))
        hidden_feat = int(args.get("hidden_feat", 64))
        out_feat = int(args.get("out_feat", 32))
        pooling = args.get("pooling", "avg")

        in_feat = int(args.get("in_feat", 113))

        model = model_cls(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            out=target_dim,
            grid_feat=grid_feat,
            num_layers=num_layers,
            pooling=pooling,
            use_bias=True,
        )

        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("KA-GNN checkpoint does not contain a valid state dict")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[KA-GNN] missing keys: {missing}")
        if unexpected:
            print(f"[KA-GNN] unexpected keys: {unexpected}")

        model.to(self._device)
        model.eval()

        self._model = model

    def _make_update_node_features(self):
        def message_func(edges):
            return {"feat": edges.data["feat"]}

        def reduce_func(nodes):
            feats = nodes.mailbox["feat"]
            count = feats.size(1) if feats.dim() >= 2 else 1
            agg = feats.sum(dim=1) / max(1, count)
            return {"agg_feats": agg}

        def updater(graph):
            graph = graph.clone()
            graph.send_and_recv(graph.edges(), message_func, reduce_func)
            agg_feats = graph.ndata.pop("agg_feats")
            graph.ndata["feat"] = self._torch.cat((graph.ndata["feat"], agg_feats), dim=1)
            return graph

        return updater

    def score(self, smiles_list: List[str]) -> np.ndarray:
        if not smiles_list:
            return np.asarray([], dtype=np.float32)

        scores = np.full(len(smiles_list), self._invalid_value, dtype=np.float32)

        graphs: List["dgl.DGLGraph"] = []
        #graphs: List["dgl.graph"] = []
        valid_indices: List[int] = []

        for idx, smiles in enumerate(smiles_list):
            if not smiles or smiles == "INVALID":
                continue
            try:
                graph = self._path_complex_mol(
                    smiles,
                    self._encoder_atom,
                    self._encoder_bond,
                    force_field=self._force_field,
                )
                if not graph or graph.num_nodes() == 0:
                    continue
                graph = self._update_node_features(graph)
                graphs.append(graph)
                valid_indices.append(idx)
            except Exception:
                continue
        print(f"Valid graphs count: {len(graphs)}")

        if not graphs:
            return scores

        batched_graph = dgl.batch(graphs).to(self._device)
        features = batched_graph.ndata["feat"].to(self._device)

        with self._torch.no_grad():
            outputs = self._model(batched_graph, features).detach().cpu().numpy()

        gap_values = outputs[:, self._gap_index]
        for local_idx, target_idx in enumerate(valid_indices):
            scores[target_idx] = float(gap_values[local_idx])

        return scores


__all__ = ["KAGnnGapPredictor"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SMILES with KA-GNN gap predictor")
    parser.add_argument("smiles", nargs="+", help="SMILES strings to score")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    predictor = KAGnnGapPredictor()
    gap_scores = predictor.score(cli_args.smiles)
    for smi, score in zip(cli_args.smiles, gap_scores.tolist()):
        print(f"{smi}\t{score}")
