from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
import copy
from typing import List, Optional, Tuple, Union


class RNN_with_latent(nn.Module):
    """
    Recurrent model with an inference-time latent variable Z that gates the RNN dynamics.

    Architecture overview
    ─────────────────────
    input_layer  : Linear(input_size → hidden_size)
    rnn_cell     : LSTMCell / GRUCell / RNNCell
    output_layer : Linear(hidden_size → output_size)
    Z (Parameter): (batch, seq_len, Z_dim) — optimized separately from weights

    Two optimizers
    ──────────────
    W_optimizer : updates all network weights via BPTT (Weight Update / WU)
    Z_optimizer : updates only Z via gradient descent (Latent Update / LU)

    Two gating modes (set in config)
    ─────────────────────────────────
    Multiplicative (use_mul_gating=True):
        A sparse random mask projects Z → gate vector (batch, hidden_size).
        gate is element-wise multiplied into the RNN hidden state before/after each step.
        This lets context modulate the RNN's dynamics without changing the inputs.

    Additive (use_add_gating=True):
        Z is concatenated with the input before the input_layer.
        Alternative to multiplicative gating; mutually exclusive in practice.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.hidden_size = int(config.hidden_size)
        self.Z_dim = int(np.prod(config.latent_dims))
        self.Z_chunks = max(int(config.latent_chunks), 1)

        if self.Z_dim % self.Z_chunks != 0:
            raise ValueError(
                f"Z_dim ({self.Z_dim}) must be divisible by latent_chunks ({self.Z_chunks})."
            )

        self.use_add_gating = bool(config.use_add_gating)
        self.use_mul_gating = bool(config.use_mul_gating)

        # Oracle latent (what_latent='context_ids'): how a raw context-id value becomes Z.
        self.oracle_context_encoding = str(getattr(config, "oracle_context_encoding", "one_hot"))
        oracle_values = getattr(config, "oracle_context_values", None)
        self.oracle_context_values = (
            None if oracle_values is None
            else torch.as_tensor(np.asarray(oracle_values, dtype=np.float32), device=self.device)
        )
        oracle_range = getattr(config, "oracle_context_range", None)
        self.oracle_context_range = (
            None if oracle_range is None else (float(oracle_range[0]), float(oracle_range[1]))
        )
        self._validate_oracle_encoding(config)

        model_input_size = config.input_size + (self.Z_dim if self.use_add_gating else 0)
        self.input_layer = nn.Linear(model_input_size, self.hidden_size)
        self.rnn_cell = self._build_recurrent_cell(config.rnn_type)
        self.output_layer = nn.Linear(self.hidden_size, config.output_size)

        if self.use_mul_gating:
            self._build_static_gates()
        else:
            self.gating_mask1 = None
            self.gating_mask2 = None

        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

        self.loss_function = nn.MSELoss(reduction="none")
        self.exponential_increase_filter: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None

        self.init_hidden()
        self.init_Z()

        self.init_exponential_increase_filter(config)
        self.W_optimizer = self._build_W_optimizer()
        self._rebuild_Z_optimizer()

        # Legacy aliases — kept for external scripts; use W_optimizer / Z_optimizer in new code
        self.WU_optimizer = self.W_optimizer
        self.LU_optimizer = self.Z_optimizer

    # ── Construction helpers ───────────────────────────────────────────────────

    def _build_recurrent_cell(self, rnn_type: str) -> nn.Module:
        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            return nn.LSTMCell(self.hidden_size, self.hidden_size)
        if rnn_type == "gru":
            return nn.GRUCell(self.hidden_size, self.hidden_size)
        if rnn_type == "rnn":
            return nn.RNNCell(self.hidden_size, self.hidden_size)
        raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Choose from lstm, gru, rnn.")

    def _build_static_gates(self) -> None:
        """Create two fixed Bernoulli masks (Z_dim, hidden_size) used for multiplicative gating."""
        prob = float(np.clip(self.config.P_gates_bernoulli_prob, 0.0, 1.0))
        shape = (self.Z_dim, self.hidden_size)
        gate1 = torch.bernoulli(torch.full(shape, prob, device=self.device))
        gate2 = torch.bernoulli(torch.full(shape, prob, device=self.device))
        self.register_buffer("gating_mask1", gate1)
        self.register_buffer("gating_mask2", gate2)
        self.P = self.gating_mask1  # legacy alias used in logging

    # ── Hidden state initialisation ────────────────────────────────────────────

    def init_hidden(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = batch_size or int(self.config.batch_size)
        # Non-zero init keeps multiplicative gates active from the first step
        self.hidden_state = torch.full((batch_size, self.hidden_size), 10.0 / self.hidden_size, device=self.device)
        if self.config.rnn_type.lower() == "lstm":
            self.cell_state = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            self.cell_state = None
        return self.hidden_state, self.cell_state

    # ── Latent variable Z lifecycle ────────────────────────────────────────────

    def init_Z(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None) -> torch.Tensor:
        """Allocate Z as zeros and rebuild the Z optimizer.

        Note: with latent_activation='none', Z=0 makes the multiplicative gate zero, which
        silences the hidden state until LU moves Z away — a batch or two with Adam, but forever
        for a condition that runs with LU off.
        """
        batch_size = batch_size or int(self.config.batch_size)
        seq_len = seq_len or int(self.config.seq_len)
        self._set_Z_parameter(torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device))
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()
        return self.Z

    def reset_Z(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None) -> None:
        """Zero Z in-place (or reallocate if shape changed). Use when pass_previous_latent=False."""
        batch_size = batch_size or int(self.config.batch_size)
        seq_len = seq_len or int(self.config.seq_len)
        if self.Z.shape == (batch_size, seq_len, self.Z_dim):
            with torch.no_grad():
                self.Z.zero_()
            self.Z.requires_grad_(True)
            self.latent = self.Z
            self.init_exponential_increase_filter(seq_len=seq_len)
            return
        self._set_Z_parameter(torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device))
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()

    def detach_Z(self) -> None:
        """Disconnect Z from the computation graph while keeping its value.
        Use when pass_previous_latent=True so the next batch's LU warm-starts from the current Z."""
        self.Z.detach_()
        self.Z.requires_grad_(True)
        self.latent = self.Z

    def set_Z(self, Z: torch.Tensor) -> None:
        """Copy an external tensor into Z (e.g., for controlled experiments)."""
        tensor = Z.to(self.device)
        if tensor.shape == self.Z.shape:
            with torch.no_grad():
                self.Z.copy_(tensor)
            self.Z.requires_grad_(True)
            self.latent = self.Z
            self.init_exponential_increase_filter(seq_len=tensor.shape[1])
            return
        self._set_Z_parameter(tensor.detach())
        self.init_exponential_increase_filter(seq_len=tensor.shape[1])
        self._rebuild_Z_optimizer()

    def _set_Z_parameter(self, tensor: torch.Tensor) -> None:
        self.Z = nn.Parameter(tensor, requires_grad=True)
        self.latent = self.Z  # legacy alias

    def _ensure_Z_shape(self, batch_size: int, seq_len: int) -> None:
        """Resize Z to (batch_size, seq_len, Z_dim), preserving existing values.
        Call this at batch start; do NOT call from inside the forward pass."""
        if self.Z.shape[0] == batch_size and self.Z.shape[1] == seq_len:
            return
        with torch.no_grad():
            tensor = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
            mb = min(batch_size, self.Z.shape[0])
            ms = min(seq_len, self.Z.shape[1])
            if mb > 0 and ms > 0:
                tensor[:mb, :ms] = self.Z[:mb, :ms]
        self._set_Z_parameter(tensor)
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()

    def _rebuild_Z_optimizer(self) -> None:
        self.Z_optimizer = self._build_Z_optimizer()
        self.LU_optimizer = self.Z_optimizer

    # ── Optimisers ─────────────────────────────────────────────────────────────

    def _build_W_optimizer(self):
        params = [p for name, p in self.named_parameters() if name != "Z"]
        lr = float(self.config.WU_lr)
        weight_decay = float(self.config.W_l2_loss or 0.0)
        opt_name = self.config.WU_optimizer.lower()
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=float(self.config.WU_momentum), weight_decay=weight_decay)
        raise ValueError(f"Unsupported WU_optimizer '{self.config.WU_optimizer}'.")

    def _z_decay_mode(self) -> str:
        """Which path applies Z_decay. See Config.Z_decay_mode; defaults to the legacy 'both'."""
        mode = str(getattr(self.config, "Z_decay_mode", "both"))
        if mode not in ("grad", "optimizer", "both"):
            raise ValueError(f"Unsupported Z_decay_mode '{mode}'. "
                             "Choose 'grad', 'optimizer' or 'both'.")
        return mode

    def _build_Z_optimizer(self):
        lr = float(self.config.Z_lr)
        opt_name = self.config.LU_optimizer.lower()
        # Under 'grad' the decay is added to the gradient in _apply_chunk_lr_and_decay instead;
        # passing it here as well is what made the effective decay 2 * Z_decay.
        decay = (float(self.config.Z_decay or 0.0)
                 if self._z_decay_mode() in ("optimizer", "both") else 0.0)
        if opt_name in ("adam", "adamw"):
            betas = self.config.LU_Adam_betas
            cls = torch.optim.Adam if opt_name == "adam" else torch.optim.AdamW
            return cls([self.Z], lr=lr, betas=betas, weight_decay=decay)
        if opt_name == "sgd":
            return torch.optim.SGD([self.Z], lr=lr, momentum=float(self.config.LU_momentum), weight_decay=decay)
        raise ValueError(f"Unsupported LU_optimizer '{self.config.LU_optimizer}'.")

    # ── Latent activation ──────────────────────────────────────────────────────

    def latent_activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation to a latent tensor. Called before any use of Z values."""
        activation = self.config.latent_activation
        if activation == "softmax":
            return torch.softmax(x / float(self.config.softmax_temp), dim=-1)
        if activation == "softmax_chunked":
            chunk_size = self.Z_dim // self.Z_chunks
            chunks = torch.split(x, chunk_size, dim=-1)
            return torch.cat([torch.softmax(c, dim=-1) for c in chunks], dim=-1)
        if activation == "sigmoid":
            return torch.sigmoid(x)
        if activation == "none":
            return x
        raise ValueError(f"Unsupported latent_activation '{activation}'.")

    # ── Latent slice helpers ───────────────────────────────────────────────────

    def _get_Z_slice(self, seq_step: int, batch_size: int, what_latent: str,
                     taskID: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return the ready-to-use latent vector for timestep seq_step. Shape: (batch, Z_dim).
        Activation is applied here so all downstream consumers receive activated values —
        except the continuous oracle encodings, which are already in the activated range
        (see _encode_context_ids).
        """
        if what_latent == "self":
            return self.latent_activation_function(self.Z[:, seq_step, :])

        if what_latent in ("uniform", "init"):
            value = 1.0 / self.Z_dim
            raw = torch.full((batch_size, self.Z_dim), value, device=self.device)
            return self.latent_activation_function(raw)

        if what_latent == "zeros":
            raw = torch.zeros(batch_size, self.Z_dim, device=self.device)
            return self.latent_activation_function(raw)

        if what_latent == "context_ids":
            if taskID is None:
                raise ValueError("taskID must be provided when what_latent='context_ids'.")
            ids = taskID.to(self.device)
            ids = ids if ids.dim() == 1 else ids[:, seq_step]
            if ids.dim() == 1:
                ids = ids.unsqueeze(-1)          # (batch, 1)
            return self._encode_context_ids(ids)

        raise ValueError(f"Unsupported what_latent '{what_latent}'.")

    def _prepare_latent_tensor(self, batch_size: int, seq_len: int, what_latent: str,
                                taskID: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return the full activated latent tensor for the entire sequence. Shape: (batch, seq_len, Z_dim).
        Used by combine_input_with_latent (additive gating path).
        """
        what_latent = what_latent or "self"
        if what_latent == "self":
            return self.latent_activation_function(self.Z[:, :seq_len, :])

        if what_latent in ("uniform", "init"):
            value = 1.0 / self.Z_dim
            raw = torch.full((batch_size, seq_len, self.Z_dim), value, device=self.device)
            return self.latent_activation_function(raw)

        if what_latent == "zeros":
            raw = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
            return self.latent_activation_function(raw)

        if what_latent == "context_ids":
            if taskID is None:
                raise ValueError("taskID must be provided when what_latent='context_ids'.")
            ids = taskID.to(self.device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(1).expand(-1, seq_len)
            if ids.dim() == 2:
                ids = ids.unsqueeze(-1)          # (batch, seq_len, 1)
            return self._encode_context_ids(ids)

        raise ValueError(f"Unsupported what_latent '{what_latent}'.")

    def _validate_oracle_encoding(self, config) -> None:
        """Check the oracle encoding against Z_dim at construction, not mid-training."""
        if getattr(config, "what_latent_to_use", "self") != "context_ids":
            return
        enc = self.oracle_context_encoding
        if enc == "one_hot":
            n = self.Z_dim if self.oracle_context_values is None else len(self.oracle_context_values)
            if n > self.Z_dim:
                raise ValueError(
                    f"oracle_context_encoding='one_hot' needs one Z slot per context: "
                    f"{n} values in config.oracle_context_values but Z_dim={self.Z_dim}. "
                    f"Set config.latent_dims=[{n}]."
                )
        elif enc in ("normalized", "circular"):
            expected = 1 if enc == "normalized" else 2
            if self.Z_dim != expected:
                raise ValueError(
                    f"oracle_context_encoding='{enc}' produces a {expected}-dim latent but "
                    f"Z_dim={self.Z_dim}. Set config.latent_dims=[{expected}]."
                )
            if self.oracle_context_range is None:
                raise ValueError(
                    f"oracle_context_encoding='{enc}' needs config.oracle_context_range=(lo, hi) "
                    f"giving the span of the context variable."
                )
            lo, hi = self.oracle_context_range
            if hi <= lo:
                raise ValueError(f"config.oracle_context_range must have hi > lo, got {(lo, hi)}.")
        else:
            raise ValueError(
                f"Unsupported oracle_context_encoding '{enc}'. "
                f"Choose from 'one_hot', 'normalized', 'circular'."
            )

    def _encode_context_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Encode raw context-id values as an oracle latent: (..., 1) values → (..., Z_dim).

        'one_hot' gives each context its own slot and carries identity only. The continuous
        encodings instead preserve the metric of the context variable, so contexts the model
        never saw still land in the right place — the point of comparing them.

        Only 'one_hot' passes through latent_activation_function. The continuous encodings are
        already in the model's post-activation range and a softmax would flatten them (over one
        dim it returns a constant 1.0, erasing the signal entirely).
        """
        enc = self.oracle_context_encoding

        if enc == "one_hot":
            slots = self._context_ids_to_slots(ids)
            latent = torch.zeros(*ids.shape[:-1], self.Z_dim, device=self.device)
            latent.scatter_(dim=-1, index=slots, value=1.0)
            return self.latent_activation_function(latent)

        lo, hi = self.oracle_context_range
        unit = (ids - lo) / (hi - lo)            # (..., 1): lo → 0, hi → 1

        if enc == "normalized":
            return unit
        if enc == "circular":
            theta = 2 * np.pi * unit
            return torch.cat([(1 + torch.cos(theta)) / 2, (1 + torch.sin(theta)) / 2], dim=-1)

        raise ValueError(f"Unsupported oracle_context_encoding '{enc}'.")

    def _context_ids_to_slots(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Map raw context-id values to one-hot slot indices in [0, Z_dim). Shape is preserved.

        With config.oracle_context_values set, each id is matched to the nearest entry in that
        table and takes that entry's position as its slot — this is what continuous context_ids
        (e.g. rotation angles in radians) need. Without a table, the id is truncated to an int,
        which is only correct when context_ids are already 0..Z_dim-1 class labels.
        """
        table = self.oracle_context_values
        if table is None:
            slots = ids.long()
        else:
            slots = (ids.unsqueeze(-1) - table).abs().argmin(dim=-1)

        if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= self.Z_dim):
            hint = (f"{len(table)} oracle_context_values are defined" if table is not None else
                    "no config.oracle_context_values table is set, so raw context_ids are used as "
                    "slot indices (correct only for integer class labels)")
            raise ValueError(
                f"what_latent='context_ids' produced slot index {int(slots.max())} but Z_dim="
                f"{self.Z_dim} ({hint}). Set config.latent_dims so Z_dim covers every context."
            )
        return slots


    def combine_input_with_latent(self, input: torch.Tensor, what_latent: str = "self",
                                   taskID: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Concatenate activated Z to input along the feature dimension (additive gating path)."""
        batch_size, seq_len, _ = input.shape
        latent = self._prepare_latent_tensor(batch_size, seq_len, what_latent, taskID)
        processed_input = input
        return torch.cat((processed_input, latent), dim=-1)

    # ── Multiplicative gating ──────────────────────────────────────────────────

    def _project_gates(self, activated_latent_slice: torch.Tensor, stage: str) -> torch.Tensor:
        """Project an already-activated latent slice through the gating mask.
        Returns gate vector of shape (batch, hidden_size)."""
        mask = self.gating_mask1 if stage == "pre" else self.gating_mask2
        if mask is None:
            raise RuntimeError("Multiplicative gating requested but masks were not initialised.")
        return torch.matmul(activated_latent_slice, mask)

    def apply_mul_gating(self, hidden_state: torch.Tensor, cell_state: Optional[torch.Tensor],
                          seq_step: int, stage: str = "pre", what_latent: str = "self",
                          taskID: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gate the hidden (and cell) state at a single timestep.

        The gate vector is computed from Z[:, seq_step, :] projected through a sparse mask.
        Z must already have the correct shape; call init_Z() or reset_Z() before forward().
        """
        if seq_step >= self.Z.shape[1]:
            raise IndexError(
                f"seq_step {seq_step} is out of range for Z with shape {self.Z.shape}. "
                "Call init_Z(batch_size, seq_len) or reset_Z() before each forward pass."
            )
        # _get_Z_slice already applies activation
        latent_slice = self._get_Z_slice(seq_step, hidden_state.shape[0], what_latent, taskID)
        gates = self._project_gates(latent_slice, stage)
        hidden_state = hidden_state * gates
        if cell_state is not None:
            cell_state = cell_state * gates
        return hidden_state, cell_state

    # ── Forward pass helpers ───────────────────────────────────────────────────

    def _rnn_step(self, x: torch.Tensor, h: torch.Tensor,
                  c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one step of the recurrent cell. Returns (h, c) for all cell types."""
        if self.config.rnn_type.lower() == "lstm":
            h, c = self.rnn_cell(x, (h, c))
        else:
            h = self.rnn_cell(x, h)
        return h, c

    def _apply_pre_gate(self, h, c, step, what_latent, taskID):
        """Apply pre-RNN multiplicative gate (no-op if pre_gating is disabled)."""
        if self.use_mul_gating and self.config.pre_gating:
            return self.apply_mul_gating(h, c, step, "pre", what_latent, taskID)
        return h, c

    def _apply_post_gate(self, h, c, step, what_latent, taskID):
        """Apply post-RNN multiplicative gate (no-op if post_gating is disabled)."""
        if self.use_mul_gating and self.config.post_gating:
            return self.apply_mul_gating(h, c, step, "post", what_latent, taskID)
        return h, c

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, input: torch.Tensor, taskID: Optional[torch.Tensor] = None,
                what_latent: str = "self"):
        """
        Run the model over a sequence.

        Args:
            input:      (batch, seq_len, input_size) — 
            taskID:     (batch, seq_len) or (batch,) — only needed when what_latent='context_ids'.
            what_latent: which latent source to use for gating ('self', 'context_ids', 'uniform', 'zeros').

        Returns:
            outputs:    list of (batch, output_size) tensors, one per timestep.
            (h, c):     final hidden and cell states.
        """
        batch_size, seq_len, _ = input.shape
        h, c = self.init_hidden(batch_size)

        processed_input = self.input_layer(input)  # (batch, seq_len, hidden_size)
        outputs: List[torch.Tensor] = []

        for step in range(seq_len):
            h, c = self._apply_pre_gate(h, c, step, what_latent, taskID)
            h, c = self._rnn_step(processed_input[:, step, :], h, c)
            h, c = self._apply_post_gate(h, c, step, what_latent, taskID)
            outputs.append(self.output_layer(h))

        return outputs, (h, c)

    # ── Gradient aggregation ───────────────────────────────────────────────────

    def adjust_Z_grads(self, apply_op: str = "average") -> None:
        """
        Aggregate Z gradients across the time dimension before the optimizer step.

        Z.grad has shape (batch, seq_len, Z_dim). Because Z is shared across timesteps,
        we collapse the time dimension before the optimizer applies its update.

        Schemes:
          'average'            — equal weight to all timesteps
          'last'               — only the final timestep's gradient
          'exponential'        — simple exponential weighting (older → less)
          'exponential_increase' — per-chunk filter with configurable steepness (recommended)
          'mask_last'          — average all timesteps except the last
          'mask_all'           — zero all gradients (disables LU for this batch)
          'none'               — leave gradients unchanged (each Z_t updated independently)
        """
        if self.Z.grad is None:
            return

        grads = self.Z.grad

        if apply_op == "average":
            mean = grads.mean(dim=1, keepdim=True)
            self.Z.grad = mean.expand_as(grads).clone()

        elif apply_op == "exponential":
            weights = torch.exp(torch.arange(grads.shape[1], device=grads.device, dtype=torch.float))
            weights = weights / weights.sum()
            weighted = (grads * weights.view(1, -1, 1)).sum(dim=1, keepdim=True)
            self.Z.grad = weighted.expand_as(grads).clone()

        elif apply_op == "last":
            last = grads[:, -1:, :]
            self.Z.grad = last.expand_as(grads).clone()

        elif apply_op == "mask_last":
            g = grads.clone()
            g[:, -1, :] = 0.0
            mean = g.mean(dim=1, keepdim=True)
            self.Z.grad = mean.expand_as(grads).clone()

        elif apply_op == "mask_all":
            self.Z.grad.zero_()

        elif apply_op == "exponential_increase":
            self._apply_exponential_increase()

        elif apply_op == "none":
            pass  # each Z_t is updated by its own gradient independently

        else:
            raise ValueError(f"Unsupported latent_aggregation_op '{apply_op}'.")

        self._apply_chunk_lr_and_decay()

    def init_exponential_increase_filter(self, config_or_seq_len=None, seq_len: Optional[int] = None) -> None:
        """
        Precompute the per-chunk exponential time-weighting filter used by 'exponential_increase'.

        Each filter is a normalized vector of length seq_len where:
            filter[t] = exp(steepness * t / seq_len)   (then normalized to sum=1)
        steepness=0 gives uniform weights; higher steepness concentrates weight on recent steps.
        """
        if isinstance(config_or_seq_len, int):
            seq_len = config_or_seq_len
        elif hasattr(config_or_seq_len, "seq_len"):
            seq_len = int(config_or_seq_len.seq_len)

        seq_len = seq_len or int(self.config.seq_len)
        if seq_len <= 0:
            self.exponential_increase_filter = None
            return

        x = torch.linspace(0, seq_len - 1, steps=seq_len, device=self.device)
        steepness_values = self.config.exponential_increase_steepness
        multipliers = self.config.exponential_increase_multipliers

        filters: List[torch.Tensor] = []
        for idx in range(self.Z_chunks):
            steepness = float(steepness_values[idx] if idx < len(steepness_values) else steepness_values[-1])
            multiplier = float(multipliers[idx] if idx < len(multipliers) else multipliers[-1])
            initial = torch.exp(torch.tensor(-steepness, device=self.device))
            rate = steepness / max(seq_len, 1)
            y = initial * torch.exp(rate * x)
            y = y / y.sum()
            filters.append(y * multiplier)

        self.exponential_increase_filter = filters[0] if self.Z_chunks == 1 else filters

    def _apply_exponential_increase(self) -> None:
        """Apply the precomputed exponential filter to Z.grad, then average across time."""
        if self.Z.grad is None or self.exponential_increase_filter is None:
            return

        grads = self.Z.grad.clone()
        if isinstance(self.exponential_increase_filter, list):
            chunk_size = self.Z_dim // self.Z_chunks
            for idx, filt in enumerate(self.exponential_increase_filter):
                filt = filt.to(grads.device)
                sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
                grads[:, :, sl] = grads[:, :, sl] * filt.view(1, -1, 1)
        else:
            filt = self.exponential_increase_filter.to(grads.device)
            grads = grads * filt.view(1, -1, 1)

        mean = grads.mean(dim=1, keepdim=True)
        self.Z.grad = mean.expand_as(self.Z.grad).clone()

    def _apply_chunk_lr_and_decay(self) -> None:
        """
        Optionally scale gradients and add L2 decay per chunk.
        Active only if config.chunk_LU_lrs or config.chunk_l2_losses are set.
        In standard runs these attributes don't exist, so this is effectively a no-op.
        """
        if self.Z.grad is None:
            return

        grad = self.Z.grad
        base_lr = float(self.config.Z_lr) or 1.0
        chunk_lrs = getattr(self.config, "chunk_LU_lrs", None)
        base_decay = float(self.config.Z_decay or 0.0)
        chunk_decays = getattr(self.config, "chunk_l2_losses", None)
        chunk_size = self.Z_dim // self.Z_chunks

        for idx in range(self.Z_chunks):
            start = idx * chunk_size
            end = min(self.Z_dim, (idx + 1) * chunk_size)
            if start >= end:
                continue

            if chunk_lrs and idx < len(chunk_lrs):
                lr_scale = float(chunk_lrs[idx]) / base_lr if base_lr != 0 else 1.0
                if lr_scale != 1.0:
                    grad[:, :, start:end] *= lr_scale

            decay = float(chunk_decays[idx]) if (chunk_decays and idx < len(chunk_decays)) else base_decay
            if decay:
                grad[:, :, start:end] += decay * self.Z[:, :, start:end]

    # ── RL helpers ────────────────────────────────────────────────────────────

    def pick_action(self, action_distribution: torch.Tensor,
                    offline_action: Optional[torch.Tensor] = None):
        categorical = torch.distributions.Categorical(action_distribution)
        action = offline_action if offline_action is not None else categorical.sample()
        return action, categorical.log_prob(action)

    # ── Legacy method aliases — use the Z_ names in new code ──────────────────

    def init_latent(self, *args, **kwargs):
        return self.init_Z(*args, **kwargs)

    def reset_latent(self, *args, **kwargs):
        return self.reset_Z(*args, **kwargs)

    def detach_latent(self):
        return self.detach_Z()

    def set_latent(self, *args, **kwargs):
        return self.set_Z(*args, **kwargs)

    def get_WU_optimizer(self):
        return self.W_optimizer

    def get_LU_optimizer(self):
        return self.Z_optimizer

    def adjust_latent_grads(self, *args, **kwargs):
        return self.adjust_Z_grads(*args, **kwargs)

    # ── Deep copy support ─────────────────────────────────────────────────────

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_model = self.__class__(copy.deepcopy(self.config, memo)).to(self.device)
        new_model.load_state_dict({k: v.clone() for k, v in self.state_dict().items()}, strict=True)
        memo[id(self)] = new_model
        return new_model
