"""Adapter wrapping a trained CTGAN model for noise-based generation."""
from __future__ import annotations

import os
import sys
from typing import Union

import numpy as np
import pandas as pd
import torch


class CTGANRepoAdapter:
    """Wrapper around a trained CTGAN for generation via direct noise injection."""

    def __init__(self, ctgan_model):
        self.m = ctgan_model

    @staticmethod
    def load(path: str, device: str = "cpu", ctgan_lib_parent: str | None = None) -> CTGANRepoAdapter:
        """Load a CTGAN .pkl checkpoint and return an adapter."""
        if ctgan_lib_parent is None:
            ctgan_lib_parent = os.path.dirname(os.path.abspath(path))
        ctgan_lib_parent = os.path.abspath(ctgan_lib_parent)
        if ctgan_lib_parent not in sys.path:
            sys.path.insert(0, ctgan_lib_parent)
        model = torch.load(path, map_location=device, weights_only=False)
        if hasattr(model, "set_device"):
            model.set_device(device)
        return CTGANRepoAdapter(model)

    @property
    def z_dim(self) -> int:
        return int(self.m._embedding_dim)

    @property
    def device(self) -> torch.device:
        return next(self.m._generator.parameters()).device

    def generate_from_noise(
        self,
        Z: Union[torch.Tensor, np.ndarray],
        cond_vec: np.ndarray | None = None,
        batch_rows: int = 4096,
    ) -> pd.DataFrame:
        """Run noise Z through the CTGAN generator -> DataFrame."""
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
        Z = np.asarray(Z, dtype=np.float32)
        assert Z.shape[1] == self.z_dim, f"Expected z_dim={self.z_dim}, got {Z.shape[1]}"
        device = self.device
        all_data = []
        for start in range(0, Z.shape[0], batch_rows):
            end = min(start + batch_rows, Z.shape[0])
            z_batch = torch.tensor(Z[start:end], dtype=torch.float32, device=device)
            if cond_vec is not None:
                cv_batch = cond_vec[start:end]
            else:
                cv_batch = self.m._data_sampler.sample_original_condvec(z_batch.shape[0])
            if cv_batch is not None:
                c = torch.from_numpy(cv_batch).to(device)
                z_input = torch.cat([z_batch, c], dim=1)
            else:
                z_input = z_batch
            with torch.no_grad():
                fake = self.m._generator(z_input)
                fakeact = self.m._apply_activate(fake)
            all_data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(all_data, axis=0)
        result = self.m._transformer.inverse_transform(data)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def sample_cond_vec(self, n: int) -> np.ndarray | None:
        """Pre-sample conditional vectors."""
        return self.m._data_sampler.sample_original_condvec(n)
