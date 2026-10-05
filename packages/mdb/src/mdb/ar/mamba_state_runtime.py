"""Resident recurrent-state consumer for Mamba-family model virtual memory.

This is the Mamba analogue of :class:`CausalContextQwenRuntime`.  Where the Qwen
runtime places one shared ancestor K/V and private deltas and steps the decoder
one token at a time, this runtime hydrates a durable per-layer ``(C_l, H_l)``
state program into a native cache and runs a declared suffix through the stock
forward and frozen LM head.  Pages are the per-layer ``MambaLayerState`` cells of
a :class:`MambaStateProgram`.

The read and write primitives follow the space/time paper's reading rule 6
(`research/papers/space-and-time-circuits/paper.md` §2.3):

- **read / ablation** happens at the *formed-circuit cut* — the recurrent state
  page.  ``stage``/``stage_batched`` install a (possibly page-swapped) formed
  state and read the native consumer.
- **writes** are offered through the *residual* at the source position;
  ``residual_write_stage`` adds the (donor - base) block-output residual at the
  carrier positions of one or more layers and lets the model form its own state.

The numeric kernels are byte-for-byte the ones validated in
``experiments/2026-09-30-mamba-full-layer-sweeps/worker_sweep.py`` (staged suffix
forward, batched per-layer sweep, residual-write sweep) and
``experiments/2026-09-30-mamba-writer-clock/worker_clock.py`` (per-layer clock),
so a VM sweep reproduces the recorded ``sweep130`` profile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch
import torch.nn.functional as F

from mdb.ar.mamba_state_program import (
    MambaLayerState,
    MambaStateProgram,
    mamba_execution_abi,
)


class MambaStateRuntimeError(ValueError):
    """A Mamba state mount, staged forward, or clock read violates its ABI."""


def mamba_backbone_layers(model: Any) -> Sequence[Any]:
    """Return the per-layer residual blocks in forward order (for residual writes)."""

    backbone = getattr(model, "backbone", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise MambaStateRuntimeError("model has no backbone.layers residual blocks")
    return layers


def mamba_mixers(model: Any) -> list[Any]:
    """Return the per-layer MambaMixer submodules in forward order (for the clock)."""

    needed = ("A_log", "dt_proj", "x_proj", "in_proj", "conv1d", "out_proj")
    mixers = [module for _, module in model.named_modules() if all(hasattr(module, a) for a in needed)]
    if not mixers:
        raise MambaStateRuntimeError("no MambaMixer submodules found")
    return mixers


def _cache_recurrent(cache: Any, layer_idx: int) -> torch.Tensor:
    container = cache.layers[layer_idx].recurrent_states
    if isinstance(container, torch.Tensor):
        return container
    if isinstance(container, Mapping):
        return container.get(0)
    return container[0]


def page_swapped_program(
    base: MambaStateProgram,
    donor: MambaStateProgram,
    layers: Sequence[int],
) -> MambaStateProgram:
    """Per-layer page swap: take ``donor``'s ``(C_l, H_l)`` at ``layers`` onto ``base``.

    This is the ``mixed()`` primitive of the full-layer sweep, promoted to a VM
    page operation.  A no-op swap (``donor is base`` or empty ``layers``) returns
    a byte-identical clone, which is the exact no-op gate.
    """

    count = len(base.layers)
    if len(donor.layers) != count:
        raise MambaStateRuntimeError("page swap requires programs with equal layer counts")
    selected = set(int(index) for index in layers)
    if any(index < 0 or index >= count for index in selected):
        raise MambaStateRuntimeError("page swap layer index is out of range")
    cells = []
    for index in range(count):
        source = donor if index in selected else base
        cells.append(
            MambaLayerState(
                source.layers[index].conv.clone(),
                source.layers[index].recurrent.clone(),
            )
        )
    return MambaStateProgram(
        model_id=base.model_id,
        revision=base.revision,
        source_token_ids=base.source_token_ids,
        pending_token_id=base.pending_token_id,
        layers=tuple(cells),
        model_config_sha256=base.model_config_sha256,
        tokenizer_sha256=base.tokenizer_sha256,
        source_program_fingerprint=base.source_program_fingerprint,
        execution_abi_sha256=base.execution_abi_sha256,
    )


@dataclass(slots=True)
class MambaStateDiff:
    """Per-layer state deltas between two programs (the ``diff`` page op)."""

    conv_l2: tuple[float, ...]
    recurrent_l2: tuple[float, ...]
    conv_max_abs: tuple[float, ...]
    recurrent_max_abs: tuple[float, ...]
    identical_layers: tuple[bool, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "conv_l2": list(self.conv_l2),
            "recurrent_l2": list(self.recurrent_l2),
            "conv_max_abs": list(self.conv_max_abs),
            "recurrent_max_abs": list(self.recurrent_max_abs),
            "identical_layers": list(self.identical_layers),
            "changed_layers": [i for i, same in enumerate(self.identical_layers) if not same],
        }


def diff_programs(left: MambaStateProgram, right: MambaStateProgram) -> MambaStateDiff:
    if len(left.layers) != len(right.layers):
        raise MambaStateRuntimeError("diff requires programs with equal layer counts")
    conv_l2, rec_l2, conv_mx, rec_mx, same = [], [], [], [], []
    for a, b in zip(left.layers, right.layers, strict=True):
        dc = (a.conv.float() - b.conv.float())
        dr = (a.recurrent.float() - b.recurrent.float())
        conv_l2.append(round(float(dc.norm()), 8))
        rec_l2.append(round(float(dr.norm()), 8))
        conv_mx.append(round(float(dc.abs().max()), 8))
        rec_mx.append(round(float(dr.abs().max()), 8))
        same.append(bool(dc.abs().max() == 0.0 and dr.abs().max() == 0.0))
    return MambaStateDiff(tuple(conv_l2), tuple(rec_l2), tuple(conv_mx), tuple(rec_mx), tuple(same))


class MambaStateRuntime:
    """A resident exact-replay consumer for one Mamba model and its state pages."""

    def __init__(self, model: Any, *, model_abi_sha256: str) -> None:
        if len(model_abi_sha256) != 64:
            raise MambaStateRuntimeError("model ABI must be a SHA-256 digest")
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_abi_sha256 = model_abi_sha256
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.execution_abi = mamba_execution_abi(model.config)
        self.execution_abi_sha256 = self.execution_abi["fingerprint"]
        self.n_layers = len(mamba_backbone_layers(model))

    # -- capture (export) -------------------------------------------------
    @torch.inference_mode()
    def capture(
        self,
        token_ids: Sequence[int],
        *,
        cut_pos: int,
        pending_token_id: int,
        model_id: str,
        revision: str,
    ) -> MambaStateProgram:
        """Compile the prefix ``token_ids[: cut_pos + 1]`` into a durable state program."""

        ids = torch.tensor([[int(t) for t in token_ids[: cut_pos + 1]]], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=True, return_dict=True)
        cache = getattr(out, "cache_params", None)
        if cache is None:
            cache = out.past_key_values
        return MambaStateProgram.from_cache(
            cache=cache,
            model_id=model_id,
            revision=revision,
            source_token_ids=tuple(int(t) for t in token_ids[: cut_pos + 1]),
            pending_token_id=int(pending_token_id),
            execution_abi_sha256=self.execution_abi_sha256,
        )

    def _assert_abi(self, program: MambaStateProgram) -> None:
        if program.execution_abi_sha256 not in (None, self.execution_abi_sha256):
            raise MambaStateRuntimeError("program execution ABI changed; recurrent state needs recompilation")

    # -- formed-state read (staged suffix forward) ------------------------
    @torch.inference_mode()
    def stage(self, program: MambaStateProgram, suffix: Sequence[int]) -> torch.Tensor:
        """Install one formed state page set and read the native consumer over ``suffix``."""

        self._assert_abi(program)
        cache = program.hydrate(
            self.model.config, device=self.device, dtype=self.dtype, preserve_state_dtype=True
        )
        log_probs = None
        for token in suffix:
            out = self.model(
                input_ids=torch.tensor([[int(token)]], dtype=torch.long, device=self.device),
                cache_params=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.cache_params
            log_probs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        if log_probs is None:
            raise MambaStateRuntimeError("staged read requires a non-empty suffix")
        return log_probs

    def _hydrate_batch(self, programs: Sequence[MambaStateProgram]) -> Any:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.model.config)
        for index in range(self.n_layers):
            conv = torch.cat([p.layers[index].conv.to(self.device) for p in programs], dim=0)
            rec = torch.cat([p.layers[index].recurrent.to(self.device) for p in programs], dim=0)
            layer = cache.layers[index]
            layer.update_conv_state(conv, state_idx=0, conv_kernel_size=int(conv.shape[-1]))
            layer.update_recurrent_state(rec, state_idx=0)
        return cache

    @torch.inference_mode()
    def stage_batched(
        self, programs: Sequence[MambaStateProgram], suffix: Sequence[int], *, chunk: int = 32
    ) -> torch.Tensor:
        """Batched formed-state read: one row per program, shared suffix (``staged_logits_batched``)."""

        for program in programs:
            self._assert_abi(program)
        rows = []
        for start in range(0, len(programs), chunk):
            batch = programs[start : start + chunk]
            cache = self._hydrate_batch(batch)
            log_probs = None
            for token in suffix:
                tokens = torch.full((len(batch), 1), int(token), dtype=torch.long, device=self.device)
                out = self.model(input_ids=tokens, cache_params=cache, use_cache=True, return_dict=True)
                cache = out.cache_params
                log_probs = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
            rows.append(log_probs.detach())
        return torch.cat(rows, dim=0)

    @torch.inference_mode()
    def native_full(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Native full-sequence forward final-token log-probs (replay-parity reference)."""

        ids = torch.tensor([[int(t) for t in token_ids]], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=False, return_dict=True)
        return torch.log_softmax(out.logits[0, -1].float(), dim=-1)

    # -- residual write (write through the residual at the source) --------
    @torch.inference_mode()
    def block_output_residual_deltas(
        self, base_ids: Sequence[int], donor_ids: Sequence[int], positions: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        """(donor - base) block-output residual at ``positions`` for every layer."""

        blocks = mamba_backbone_layers(self.model)
        n = len(blocks)

        def block_outputs(ids: Sequence[int]) -> dict[int, torch.Tensor]:
            outs: dict[int, torch.Tensor] = {}
            handles = []
            for index, block in enumerate(blocks):
                def make(index):
                    def hook(_module, _inp, out):
                        tensor = out[0] if isinstance(out, tuple) else out
                        outs[index] = tensor.detach().clone()
                    return hook
                handles.append(block.register_forward_hook(make(index)))
            try:
                self.model(
                    input_ids=torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=self.device),
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            return outs

        base_out = block_outputs(base_ids)
        donor_out = block_outputs(donor_ids)
        pidx = torch.tensor([int(p) for p in positions], device=self.device)
        return {index: (donor_out[index][0, pidx] - base_out[index][0, pidx]) for index in range(n)}

    @torch.inference_mode()
    def residual_write_stage(
        self,
        base_ids: Sequence[int],
        positions: Sequence[int],
        residual_deltas: Mapping[int, torch.Tensor],
        layer_rows: Sequence[Sequence[int]],
        *,
        chunk: int = 32,
    ) -> torch.Tensor:
        """Row ``r`` adds the residual delta at every layer in ``layer_rows[r]`` (``resid_write_logits``)."""

        blocks = mamba_backbone_layers(self.model)
        n = len(blocks)
        pidx = torch.tensor([int(p) for p in positions], device=self.device)
        base = torch.tensor([[int(t) for t in base_ids]], dtype=torch.long, device=self.device)
        rows_out = []
        for start in range(0, len(layer_rows), chunk):
            rows = layer_rows[start : start + chunk]
            batch = base.repeat(len(rows), 1)
            handles = []
            for index in range(n):
                selected = [r for r, layers in enumerate(rows) if index in layers]
                if not selected:
                    continue

                def make(index, selected):
                    sel = torch.tensor(selected, device=self.device)
                    delta = residual_deltas[index]

                    def hook(_module, _inp, out):
                        tensor = out[0] if isinstance(out, tuple) else out
                        tensor = tensor.clone()
                        tensor[sel[:, None], pidx[None, :]] += delta.to(tensor.dtype)
                        return (tensor,) + tuple(out[1:]) if isinstance(out, tuple) else tensor

                    return hook

                handles.append(blocks[index].register_forward_hook(make(index, selected)))
            try:
                out = self.model(input_ids=batch, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            rows_out.append(torch.log_softmax(out.logits[:, -1].float(), dim=-1))
        return torch.cat(rows_out, dim=0)

    # -- clock (per-layer retention + write magnitude) --------------------
    @torch.inference_mode()
    def layer_clock(self, span_ids: Sequence[int], cut_pos: int) -> dict[str, Any]:
        """Per-layer slow-path clock over ``span_ids`` with subject cut at ``cut_pos``.

        Reuses the exact scan math of
        ``experiments/2026-09-30-mamba-writer-clock/worker_clock.py``:
        retention ``R = exp(A * sum_suffix dt)`` and the subject write
        ``deltaB_u[:, :, cut_pos, :]``.  Returns per-layer ``surv_norm_frac``
        (fraction of cut-time state norm surviving to the end), ``ret_mean``,
        ``write_norm``, ``hs_norm`` and a canary against the native cache state.
        """

        mixers = mamba_mixers(self.model)
        n = len(mixers)
        ids = torch.tensor([[int(t) for t in span_ids]], dtype=torch.long, device=self.device)
        seq_len = ids.shape[1]
        if not 0 <= cut_pos < seq_len - 1:
            raise MambaStateRuntimeError("clock requires a subject cut with at least one suffix token")

        captured: dict[int, torch.Tensor] = {}
        handles = []
        for i, mixer in enumerate(mixers):
            def pre(_module, args, kwargs, idx=i):
                captured[idx] = (args[0] if args else kwargs["input_states"]).detach()
            handles.append(mixer.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            self.model(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

        pre_out = self.model(input_ids=ids[:, : cut_pos + 1], use_cache=True, return_dict=True)
        native_cache = getattr(pre_out, "cache_params", None) or pre_out.past_key_values

        surv, ret_mean, ret_median, write_norm, hs_norm, cumdt_mean = [], [], [], [], [], []
        canary_max_abs = 0.0
        for i, mixer in enumerate(mixers):
            inp = captured[i].to(device=self.device, dtype=self.dtype)
            seq = inp.shape[1]
            din = int(mixer.intermediate_size)
            ds = int(mixer.ssm_state_size)
            tsr = int(mixer.time_step_rank)
            proj = mixer.in_proj(inp).transpose(1, 2)
            hs, _gate = proj.chunk(2, dim=1)
            hs = mixer.act(mixer.conv1d(hs)[..., :seq])
            ssm_params = mixer.x_proj(hs.transpose(1, 2))
            time_step, b_mat, _c_mat = torch.split(ssm_params, [tsr, ds, ds], dim=-1)
            dt = F.softplus(mixer.dt_proj(time_step)).transpose(1, 2)
            a_mat = -torch.exp(mixer.A_log.float())
            d_a = torch.exp(a_mat[None, :, None, :] * dt[:, :, :, None])
            d_b = dt[:, :, :, None] * b_mat[:, None, :, :].float()
            delta_b_u = d_b * hs[:, :, :, None].float()

            state = torch.zeros(1, din, ds, device=self.device, dtype=torch.float32)
            for t in range(cut_pos + 1):
                state = d_a[:, :, t, :] * state + delta_b_u[:, :, t, :]
            h_s = state[0]
            try:
                native = _cache_recurrent(native_cache, i)[0].float()
                canary_max_abs = max(canary_max_abs, float((h_s - native).abs().max()))
            except Exception:
                pass

            write_s = delta_b_u[0, :, cut_pos, :]
            cumdt = dt[0, :, cut_pos + 1 :].sum(dim=-1)
            retention = torch.exp(a_mat * cumdt[:, None])
            hn = float(h_s.norm())
            surv.append(round(float((h_s * retention).norm()) / hn if hn > 0 else 0.0, 6))
            ret_mean.append(round(float(retention.mean()), 6))
            ret_median.append(round(float(retention.median()), 6))
            write_norm.append(round(float(write_s.norm()), 6))
            hs_norm.append(round(hn, 6))
            cumdt_mean.append(round(float(cumdt.mean()), 6))

        return {
            "schema": "saturn-mamba-vm-clock-v1",
            "n_layers": n,
            "cut_pos": int(cut_pos),
            "suffix_len": seq_len - 1 - int(cut_pos),
            "canary_recurrent_max_abs": round(canary_max_abs, 8),
            "surv_norm_frac": surv,
            "ret_mean": ret_mean,
            "ret_median": ret_median,
            "write_norm": write_norm,
            "hs_norm": hs_norm,
            "cumdt_mean": cumdt_mean,
        }


def logits_sha256(log_probs: torch.Tensor) -> str:
    raw = log_probs.detach().float().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return sha256(raw).hexdigest()
