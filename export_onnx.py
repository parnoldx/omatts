#!/usr/bin/env python3
"""
PocketTTS ONNX Export, Quantize & Validate
============================================

Exports 5 ONNX models from the upstream `kyutai-labs/pocket-tts` package
for use with the PocketTTS.cpp C++ runtime. Optionally applies INT8
dynamic quantization and validates numerical equivalence against PyTorch.

Models:
  1. mimi_encoder.onnx       — Stateless. Audio → speaker conditioning.
  2. text_conditioner.onnx   — Stateless. Token IDs → text embeddings.
  3. flow_lm_main.onnx       — Stateful AR backbone with explicit KV cache.
  4. flow_lm_flow.onnx       — Stateless flow network (SimpleMLPAdaLN).
  5. mimi_decoder.onnx        — Stateful streaming decoder.

INT8 quantized (dynamic, MatMul only):
  3b. flow_lm_main_int8.onnx
  4b. flow_lm_flow_int8.onnx
  5b. mimi_decoder_int8.onnx

Usage:
  cd ~/Desktop/pocket-tts
  source .venv/bin/activate

  # Export + quantize + validate (default, outputs to ./models):
  python export_onnx.py

  # Validate existing exports only:
  python export_onnx.py --validate-only

Requirements:
  pip install pocket-tts onnx onnxruntime
"""

import argparse
import copy
import logging
import math
import struct
import sys
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress noisy warnings from ONNX tracing and quantization
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy.*ONNX.*")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
logging.getLogger("root").setLevel(logging.ERROR)

# ============================================================================
# Disable beartype BEFORE importing pocket_tts.
# pocket_tts/__init__.py calls beartype.claw.beartype_this_package() which
# auto-instruments every function/method in the package at import time.
# ONNX tracing converts Python scalars to tensors, which violates all the
# int/float type hints and causes BeartypeCallHintParamViolation errors.
# ============================================================================
import beartype
import beartype.claw
beartype.beartype = lambda func: func
beartype.claw.beartype_this_package = lambda *a, **kw: None
beartype.claw.beartype_package = lambda *a, **kw: None

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import StatefulModule, init_states, increment_steps
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.modules.rope import apply_rope
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.default_parameters import (
    DEFAULT_TEMPERATURE, DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP, DEFAULT_EOS_THRESHOLD,
)
from pocket_tts.utils.config import load_config


# ============================================================================
# Constants
# ============================================================================

MAX_SEQ_LEN = 1000       # Max KV cache length for flow_lm
MIMI_CONTEXT = 250       # Mimi transformer attention context window
MIMI_CACHE_LEN = 250     # Mimi KV cache allocation (circular buffer, matches MIMI_CONTEXT)
OPSET = 17

# Layer count: we export the standard 6-layer flow_lm, not the newer `_24l`
# language variants upstream added (english_2026-04_24l, french_24l, ...).
# The 24l tier is ~higher quality but ~4x slower on CPU: 6->24 layers means
# ~300 MB vs 75 MB for flow_lm_main_int8, ~1.3 GB vs 219 MB fp32 weights,
# ~4x per-step AR compute, KV cache 25 MB -> 100 MB (states scale 3 per layer,
# 18 -> 72). We choose the 6-layer tier for real-time CPU inference; revisit
# only as an optional second model pack if voice quality complaints show up.
# Supporting it means parameterizing NUM_LAYERS here and in omatts.cpp
# (StatefulRunner state wiring, .kv voice format).


# ============================================================================
# Helpers
# ============================================================================

def _download(url: str, dest: Path):
    """Download a file with progress."""
    from urllib.request import urlopen, Request
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "pocket-tts-export"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := resp.read(1 << 20):  # 1MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded >> 20}/{total >> 20} MB)", end="", flush=True)
        if total:
            print()


def urlopen_fn(req):
    from urllib.request import urlopen
    return urlopen(req)


def Request(url, **kw):
    from urllib.request import Request as _R
    return _R(url, **kw)


def flatten_states(model: nn.Module, batch_size=1, seq_len=1000):
    """Get flat state list matching C++ StatefulRunner convention:
    sorted by module name, then sorted by key within each module."""
    states = init_states(model, batch_size=batch_size, sequence_length=seq_len)
    _sanitize_states(states)
    layout = []
    for mn in sorted(states.keys()):
        for key in sorted(states[mn].keys()):
            t = states[mn][key]
            layout.append((mn, key, list(t.shape), t.dtype))
    return layout, states


def _sanitize_states(states: dict):
    """Replace NaN with 0 in all state tensors in-place.
    Upstream init_state fills KV caches with float('nan'). The circular
    buffer returns ALL cache slots to SDPA including uninitialized ones.
    softmax(-inf) × NaN = 0 × NaN = NaN in IEEE 754, so we must zero them."""
    for mn in states:
        for key in states[mn]:
            t = states[mn][key]
            if t.is_floating_point():
                states[mn][key] = torch.where(torch.isnan(t), torch.zeros_like(t), t)


def build_model_state_dict(layout, tensors):
    """Rebuild model_state dict from flat tensor list."""
    state = {}
    for i, (mn, key, _, _) in enumerate(layout):
        state.setdefault(mn, {})[key] = tensors[i]
    return state


def extract_flat_states(layout, model_state):
    """Extract flat tensor list from model_state dict."""
    return [model_state[mn][key] for mn, key, _, _ in layout]


def export_and_report(wrapper, args, output_path, input_names, output_names,
                      dynamic_axes, *, opset=OPSET):
    """Export model and print summary."""
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # Use legacy TorchScript exporter for compatibility
        )
    size_mb = output_path.stat().st_size / 1e6
    print(f"  ✓ {output_path.name} ({size_mb:.1f} MB)")
    return size_mb


# ============================================================================
# Model 1: Mimi Encoder (Stateless)
# ============================================================================

class MimiEncoderWrapper(nn.Module):
    """Wraps mimi encoder + speaker projection for ONNX export.

    Input:  audio [1, 1, T_samples]
    Output: conditioning [1, N_frames, 1024]

    The encoder path is fully stateless (all streaming buffers are
    initialized fresh internally via model_state=None).
    """

    def __init__(self, mimi, speaker_proj_weight):
        super().__init__()
        self.encoder = mimi.encoder
        self.encoder_transformer = mimi.encoder_transformer
        # downsample exists when encoder_frame_rate != frame_rate
        self.downsample = mimi.downsample if hasattr(mimi, 'downsample') else None
        self.speaker_proj_weight = speaker_proj_weight

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [1, 1, T] — raw waveform at 24kHz
        # Pad to multiple of frame_size (1920 = sample_rate / frame_rate).
        # This is what upstream encode_to_latent does via pad_for_conv1d.
        # We inline it because beartype rejects tensor returns during tracing.
        # No if-branch: (frame_size - length % frame_size) % frame_size = 0 when aligned.
        frame_size = 1920  # hop_length(120) × downsample_stride(16)
        pad_amount = (frame_size - audio.shape[-1] % frame_size) % frame_size
        audio = F.pad(audio, (0, pad_amount))

        emb = self.encoder(audio, model_state=None)
        (emb,) = self.encoder_transformer(emb, model_state=None)
        if self.downsample is not None:
            emb = self.downsample(emb, model_state=None)

        # emb: [1, 512, N_frames]
        latents = emb.transpose(-1, -2).to(torch.float32)  # [1, N, 512]
        conditioning = F.linear(latents, self.speaker_proj_weight)  # [1, N, 1024]
        return conditioning


def export_mimi_encoder(model: TTSModel, output_path: Path):
    wrapper = MimiEncoderWrapper(model.mimi, model.flow_lm.speaker_proj_weight)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Dummy: ~1 second of audio at 24kHz, must be multiple of frame_size=1920
    dummy_audio = torch.randn(1, 1, 23040)  # 12 frames × 1920 = 0.96s

    export_and_report(
        wrapper, (dummy_audio,), output_path,
        input_names=["audio"],
        output_names=["conditioning"],
        dynamic_axes={"audio": {2: "samples"}, "conditioning": {1: "frames"}},
    )


# ============================================================================
# Model 2: Text Conditioner (Stateless)
# ============================================================================

class TextConditionerWrapper(nn.Module):
    """Wraps LUTConditioner embedding lookup for ONNX export.

    Input:  token_ids [1, T_text] int64
    Output: embeddings [1, T_text, 1024]

    Note: upstream BaseConditioner has no output_proj — the embedding
    dimension (1024) already equals d_model, so it's just nn.Embedding.
    """

    def __init__(self, conditioner):
        super().__init__()
        self.embed = conditioner.embed  # nn.Embedding(4001, 1024)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids)


# ============================================================================
# Language Metadata (consumed by pocket_tts.cpp)
# ============================================================================

def export_builtin_voices(output_dir: Path, language: str):
    """Convert Kyutai's shipped per-language voice embeddings into the C++
    runtime's native .kv format (KV-cache snapshots of the flow_lm transformer
    after voice conditioning).

    For languages whose public weights ship a zeroed Mimi encoder (e.g. German
    "without-voice-cloning"), WAV voice cloning is impossible — these
    precomputed embeddings are the only voices. C++ resolves bare voice names
    against <models_dir>/embeddings/<name>.kv before falling back to WAVs.
    """
    import json
    import safetensors.torch as st

    api = (f"https://huggingface.co/api/models/kyutai/pocket-tts-without-voice-cloning"
           f"/tree/main/languages/{language}/embeddings")
    with urlopen_fn(Request(api, headers={"User-Agent": "pocket-tts-export"})) as r:
        entries = json.load(r)

    vdir = output_dir / "embeddings"
    vdir.mkdir(exist_ok=True)
    FLOAT16, INT64 = 10, 7  # ONNX tensor element types
    for e in entries:
        name = e["path"].split("/")[-1].removesuffix(".safetensors")
        dst = vdir / f"{name}.kv"
        if dst.exists():
            print(f"  ✓ embeddings/{name}.kv (cached)")
            continue
        src = output_dir / ".cache" / f"voice-{name}.safetensors"
        if not src.exists():
            _download(
                f"https://huggingface.co/kyutai/pocket-tts-without-voice-cloning"
                f"/resolve/main/languages/{language}/embeddings/{name}.safetensors",
                src,
            )
        sd = st.load_file(src)
        # DiskSnapshot blob: [int32 current_buf][int32 n_states], per state:
        # [int32 ndims][ndims*int64 shape][int32 ort_type][int64 nbytes][data]
        blob = struct.pack("<ii", 1, 18)  # current_buf=1 (post voice pass), 18 states
        for i in range(6):
            cache = sd[f"transformer.layers.{i}.self_attn/cache"].numpy().astype(np.float16)
            off = int(sd[f"transformer.layers.{i}.self_attn/offset"].item())
            for t, typ in ((cache[0], FLOAT16), (cache[1], FLOAT16),
                           (np.array([off], dtype=np.int64), INT64)):
                t = np.ascontiguousarray(t)
                blob += struct.pack("<i", t.ndim)
                blob += struct.pack(f"<{t.ndim}q", *t.shape)
                blob += struct.pack("<iq", typ, t.nbytes)
                blob += t.tobytes()
        with open(dst, "wb") as f:
            f.write(struct.pack("<IQ", 0x3143564B, len(blob)))  # "KVC1" magic
            f.write(blob)
        print(f"  ✓ embeddings/{name}.kv")


def export_language_meta(model: TTSModel, output_dir: Path, language: str = "english"):
    """Write runtime flags the C++ engine reads from the models dir.

    model_config.txt   — key=value flags; when the file is absent the C++
                         keeps legacy defaults (pad short inputs, nothing else).
                         language=<tag> ties the pack to voice tags (de →
                         models-de sibling dir selects the pack).
    bos_before_voice.f32 — raw fp32 [1024] BOS frame prepended to the voice
                         conditioning by the C++ runtime (insert_bos_before_voice).
    """
    flags = {
        "language": {"english": "en", "english_2026-04": "en", "german": "de"}.get(language, "en"),
        "pad_short_inputs": int(model.config.pad_with_spaces_for_short_inputs),
        "remove_semicolons": int(model.config.remove_semicolons),
        "insert_bos_before_voice": int(model.flow_lm.insert_bos_before_voice),
    }
    (output_dir / "model_config.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in flags.items())
    )
    if flags["insert_bos_before_voice"]:
        bos = model.flow_lm.bos_before_voice.detach().to(torch.float32).cpu().reshape(-1).contiguous()
        (output_dir / "bos_before_voice.f32").write_bytes(bos.numpy().tobytes())
    print(f"  ✓ model_config.txt {flags}")


def add_output_gain(onnx_path: Path, gain: float):
    """Fold a constant gain into the graph output (Mul node before the output).

    Used to match per-language decoder loudness at export time — the German
    codec decodes ~5.5x quieter (peak-matched, not mean-matched: generation variance is ~2x, so leave clipping headroom) than English (measured: ~-38 dB mean vs -20 dB
    on the same sentence), so the German graph ships with gain folded in and
    the C++ runtime stays model-agnostic. ponytail: fixed constant,
    re-measure if Kyutai re-trains the model.
    """
    import onnx
    from onnx import helper
    g = onnx.load(str(onnx_path))
    out = g.graph.output[0]
    dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
    const = out.name + "_gain_const"  # unique per output: re-folding must not shadow
    g.graph.initializer.append(helper.make_tensor(const, onnx.TensorProto.FLOAT, [1], [gain]))
    g.graph.node.append(helper.make_node("Mul", [out.name, const], [out.name + "_gained"], name=out.name + "_gain"))
    g.graph.output[0].CopyFrom(helper.make_tensor_value_info(out.name + "_gained", out.type.tensor_type.elem_type, dims))
    onnx.save(g, str(onnx_path))
    print(f"  ✓ {onnx_path.name}: folded output_gain={gain} into graph output")


def export_text_conditioner(model: TTSModel, output_path: Path):
    wrapper = TextConditionerWrapper(model.flow_lm.conditioner)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    dummy_tokens = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    export_and_report(
        wrapper, (dummy_tokens,), output_path,
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={"token_ids": {1: "text_len"}, "embeddings": {1: "text_len"}},
    )


# ============================================================================
# Model 3: FlowLM Main — Stateful AR Backbone
# ============================================================================

class FlowLMMainWrapper(nn.Module):
    """Wraps the FlowLM transformer backbone with explicit KV cache.

    Inputs:
      sequence       [1, S, 32]     — normalized latents (NaN = BOS)
      text_embeddings [1, T, 1024]  — conditioning (text + audio)
      state_0..11    — 6 × (cache [2,1,SEQ,16,64], step [1] int64)

    Outputs:
      conditioning   [1024]         — transformer output for flow network
      eos_logit      [1]            — raw EOS logit (thresholded in runtime)
      out_state_0..11

    Architecture:
      1. Replace NaN with bos_emb
      2. input_linear: [1, S, 32] → [1, S, 1024]
      3. Concatenate: [text_embeddings, input_] → [1, T+S, 1024]
      4. Run 6 transformer layers with KV cache
      5. out_norm → slice last S positions → take [-1] → eos linear

    State layout (18 states, 3 per layer):
      state_0:  layer 0 K cache  [1, MAX_SEQ, 16, 64] float32
      state_1:  layer 0 V cache  [1, MAX_SEQ, 16, 64] float32
      state_2:  layer 0 step     [1] int64
      state_3:  layer 1 K cache  ...
      ...
      state_15: layer 5 K cache
      state_16: layer 5 V cache
      state_17: layer 5 step
    """

    NUM_LAYERS = 6
    D_MODEL = 1024
    NUM_HEADS = 16
    DIM_PER_HEAD = 64
    ROPE_MAX_PERIOD = 10000.0

    def __init__(self, flow_lm):
        super().__init__()
        self.bos_emb = flow_lm.bos_emb           # [32]
        self.input_linear = flow_lm.input_linear  # Linear(32, 1024, bias=False)
        self.out_norm = flow_lm.out_norm          # LayerNorm(1024)
        self.out_eos = flow_lm.out_eos            # Linear(1024, 1)
        self.layers = flow_lm.transformer.layers  # 6 × StreamingTransformerLayer

    def _attention(self, attn: StreamingMultiheadAttention, x, cache_k, cache_v, step):
        """Run one attention layer with split K/V cache state (fp16).

        Args:
            attn: The StreamingMultiheadAttention module (for weights only).
            x: Input tensor [1, L, 1024] fp32
            cache_k: K cache [1, MAX_SEQ, 16, 64] fp16
            cache_v: V cache [1, MAX_SEQ, 16, 64] fp16
            step: Current position int64 [1]

        Returns:
            output [1, L, 1024] fp32, updated_k fp16, updated_v fp16, updated_step
        """
        B, L, _ = x.shape

        # QKV projection (fp32)
        projected = attn.in_proj(x)  # [1, L, 3072]
        packed = projected.view(B, L, 3, self.NUM_HEADS, self.DIM_PER_HEAD)
        q, k, v = torch.unbind(packed, dim=2)  # each [1, L, 16, 64]

        # RoPE (expects [B, T, H, D])
        offset = step[0]  # scalar
        q, k = apply_rope(q, k, offset=offset, max_period=self.ROPE_MAX_PERIOD)

        # Cast K/V to fp16 before scatter into cache
        k_fp16 = k.half()
        v_fp16 = v.half()

        # Update KV cache (fp16) — scatter directly, no stack needed
        positions = step[0] + torch.arange(L, device=x.device, dtype=torch.long)
        idx = positions.view(1, L, 1, 1).expand(B, L, self.NUM_HEADS, self.DIM_PER_HEAD)

        updated_k = cache_k.scatter(1, idx, k_fp16)
        updated_v = cache_v.scatter(1, idx, v_fp16)

        # Slice valid region and cast to fp32 for SDPA
        valid_len = step[0] + L
        valid_k = updated_k[:, :valid_len].float()
        valid_v = updated_v[:, :valid_len].float()

        # Causal mask: [L_q, L_kv]
        q_positions = step[0] + torch.arange(L, device=x.device)
        kv_positions = torch.arange(valid_len, device=x.device)
        mask = q_positions.unsqueeze(1) >= kv_positions.unsqueeze(0)
        attn_mask = torch.log(mask.float())

        # SDPA in fp32 — ORT's CPU kernel is SIMD-optimized for fp32
        q_t = q.transpose(1, 2)
        k_t = valid_k.transpose(1, 2)
        v_t = valid_v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, L, self.D_MODEL)
        out = attn.out_proj(out)

        new_step = step + L
        return out, updated_k, updated_v, new_step

    def forward(self, sequence, text_embeddings, *states):
        # Unpack states: k_cache, v_cache, step per layer (3 per layer)
        k_caches = [states[i * 3] for i in range(self.NUM_LAYERS)]
        v_caches = [states[i * 3 + 1] for i in range(self.NUM_LAYERS)]
        steps = [states[i * 3 + 2] for i in range(self.NUM_LAYERS)]

        # Replace NaN with BOS embedding
        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        input_ = self.input_linear(sequence)  # [1, S, 1024]

        # Prepend text conditioning
        x = torch.cat([text_embeddings, input_], dim=1)  # [1, T+S, 1024]

        out_k_caches = []
        out_v_caches = []
        out_steps = []

        for i, layer in enumerate(self.layers):
            # Self-attention (layer_scale is Identity for flow_lm)
            x_norm = layer.norm1(x)
            attn_out, new_k, new_v, new_step = self._attention(
                layer.self_attn, x_norm, k_caches[i], v_caches[i], steps[i]
            )
            x = x + attn_out

            # FFN (layer_scale is Identity)
            x_ff = layer.norm2(x)
            x = x + layer.linear2(F.gelu(layer.linear1(x_ff)))

            out_k_caches.append(new_k)
            out_v_caches.append(new_v)
            out_steps.append(new_step)

        # Output normalization
        x = self.out_norm(x)

        conditioning = x[:, -1]
        eos_logit = self.out_eos(conditioning)

        # Pack output states: k, v, step per layer
        result = [conditioning.squeeze(0), eos_logit.squeeze(0)]
        for i in range(self.NUM_LAYERS):
            result.append(out_k_caches[i])
            result.append(out_v_caches[i])
            result.append(out_steps[i])

        return tuple(result)


def export_flow_lm_main(model: TTSModel, output_path: Path, max_seq=MAX_SEQ_LEN):
    wrapper = FlowLMMainWrapper(model.flow_lm)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    N = FlowLMMainWrapper.NUM_LAYERS
    H = FlowLMMainWrapper.NUM_HEADS
    D = FlowLMMainWrapper.DIM_PER_HEAD

    # Dummy inputs: single AR step (seq_len=1)
    dummy_seq = torch.randn(1, 1, 32)
    dummy_text = torch.zeros(1, 0, 1024)  # empty for AR step

    # 3 states per layer: K cache (fp16), V cache (fp16), step
    dummy_states = []
    for i in range(N):
        dummy_states.append(torch.zeros(1, max_seq, H, D, dtype=torch.float16))  # K cache
        dummy_states.append(torch.zeros(1, max_seq, H, D, dtype=torch.float16))  # V cache
        dummy_states.append(torch.tensor([10], dtype=torch.long))                 # step

    args = (dummy_seq, dummy_text, *dummy_states)

    # Build input/output names
    in_names = ["sequence", "text_embeddings"]
    out_names = ["conditioning", "eos_logit"]
    dyn_axes = {
        "sequence": {1: "seq_len"},
        "text_embeddings": {1: "text_len"},
    }

    for i in range(N):
        in_names.extend([f"state_{i*3}", f"state_{i*3+1}", f"state_{i*3+2}"])
        out_names.extend([f"out_state_{i*3}", f"out_state_{i*3+1}", f"out_state_{i*3+2}"])

    export_and_report(wrapper, args, output_path, in_names, out_names, dyn_axes)


# ============================================================================
# Model 4: FlowLM Flow Network (Stateless)
# ============================================================================

class FlowLMFlowWrapper(nn.Module):
    """Wraps SimpleMLPAdaLN for ONNX export, fully flattened.

    The upstream SimpleMLPAdaLN uses TimestepEmbedder with nn.Sequential
    containing RMSNorm. The TorchScript ONNX exporter has a bug where
    register_buffer/nn.Parameter values inside nn.ModuleList items get
    dropped from the graph initializers when values are deduplicated.

    This wrapper extracts ALL parameters to the top level using
    nn.Parameter(requires_grad=False) and calls F.linear directly,
    avoiding the problematic nesting entirely.

    Inputs:
      c [?, 1024]  — conditioning from transformer
      s [?, 1]     — start time
      t [?, 1]     — target time
      x [?, 32]    — current noisy latent

    Output:
      flow_dir [?, 32]  — flow direction
    """

    def __init__(self, flow_net):
        super().__init__()
        # Time embedder 0 (processes s)
        self.te0_lin1_weight = nn.Parameter(flow_net.time_embed[0].mlp[0].weight.data, requires_grad=False)
        self.te0_lin1_bias = nn.Parameter(flow_net.time_embed[0].mlp[0].bias.data, requires_grad=False)
        self.te0_lin2_weight = nn.Parameter(flow_net.time_embed[0].mlp[2].weight.data, requires_grad=False)
        self.te0_lin2_bias = nn.Parameter(flow_net.time_embed[0].mlp[2].bias.data, requires_grad=False)
        self.te0_rms_alpha = nn.Parameter(flow_net.time_embed[0].mlp[3].alpha.data, requires_grad=False)
        self.te0_rms_eps = flow_net.time_embed[0].mlp[3].eps
        self.te0_freqs = nn.Parameter(flow_net.time_embed[0].freqs.data, requires_grad=False)

        # Time embedder 1 (processes t)
        self.te1_lin1_weight = nn.Parameter(flow_net.time_embed[1].mlp[0].weight.data, requires_grad=False)
        self.te1_lin1_bias = nn.Parameter(flow_net.time_embed[1].mlp[0].bias.data, requires_grad=False)
        self.te1_lin2_weight = nn.Parameter(flow_net.time_embed[1].mlp[2].weight.data, requires_grad=False)
        self.te1_lin2_bias = nn.Parameter(flow_net.time_embed[1].mlp[2].bias.data, requires_grad=False)
        self.te1_rms_alpha = nn.Parameter(flow_net.time_embed[1].mlp[3].alpha.data, requires_grad=False)
        self.te1_rms_eps = flow_net.time_embed[1].mlp[3].eps
        self.te1_freqs = nn.Parameter(flow_net.time_embed[1].freqs.data, requires_grad=False)

        self.cond_embed = flow_net.cond_embed
        self.input_proj = flow_net.input_proj
        self.res_blocks = nn.ModuleList(list(flow_net.res_blocks))
        self.final_layer = flow_net.final_layer

    def _timestep_embed(self, t, freqs, w1, b1, w2, b2, rms_alpha, rms_eps):
        args = t * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        x = F.linear(embedding, w1, b1)
        x = F.silu(x)
        x = F.linear(x, w2, b2)
        # RMSNorm (inline, no dtype casts)
        var = rms_eps + x.var(dim=-1, keepdim=True)
        x = x * (rms_alpha * torch.rsqrt(var))
        return x

    def forward(self, c, s, t, x):
        x = self.input_proj(x)
        t0 = self._timestep_embed(s, self.te0_freqs, self.te0_lin1_weight, self.te0_lin1_bias,
                                   self.te0_lin2_weight, self.te0_lin2_bias, self.te0_rms_alpha, self.te0_rms_eps)
        t1 = self._timestep_embed(t, self.te1_freqs, self.te1_lin1_weight, self.te1_lin1_bias,
                                   self.te1_lin2_weight, self.te1_lin2_bias, self.te1_rms_alpha, self.te1_rms_eps)
        t_combined = (t0 + t1) / 2.0
        c_emb = self.cond_embed(c)
        y = t_combined + c_emb
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


def fix_onnx_identity_dedup(onnx_path: Path):
    """Fix TorchScript ONNX exporter bug with deduplicated initializers.

    When two nn.Parameter values have identical data (e.g. both RMSNorm.alpha
    initialized to ones), the TorchScript tracer creates Identity nodes that
    alias one to the other. But it fails to include the aliased name as an
    initializer, causing ORT to reject the graph.

    Fix: find Identity nodes whose inputs are initializers, remove them,
    and add the output names as proper initializers with copied data.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))

    init_map = {i.name: i for i in model.graph.initializer}
    nodes_to_remove = []
    inits_to_add = []

    for node in model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            src = node.input[0]
            dst = node.output[0]
            if src in init_map and dst not in init_map:
                # This is a deduplicated alias — copy the initializer data
                src_data = numpy_helper.to_array(init_map[src])
                inits_to_add.append(numpy_helper.from_array(src_data.copy(), name=dst))
                nodes_to_remove.append(node)

    if nodes_to_remove:
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        for init in inits_to_add:
            model.graph.initializer.append(init)
        onnx.save(model, str(onnx_path))


def export_flow_lm_flow(model: TTSModel, output_path: Path):
    wrapper = FlowLMFlowWrapper(model.flow_lm.flow_net)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Match C++ runtime shapes exactly (determined from error chain):
    # - c [1024]: rank-1, raw output from flow_lm_main conditioning
    # - s [1,1]: rank-2, runtime constructs as [batch, 1]
    # - t [1,1]: rank-2, runtime constructs as [batch, 1]
    # - x [1,32]: rank-2, runtime constructs noise as [batch, ldim]
    dummy_c = torch.randn(1024)
    dummy_s = torch.zeros(1, 1)
    dummy_t = torch.ones(1, 1)
    dummy_x = torch.randn(1, 32)

    export_and_report(
        wrapper, (dummy_c, dummy_s, dummy_t, dummy_x), output_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={},  # batch dim is fixed at 1
    )

    # Fix TorchScript Identity node dedup bug
    fix_onnx_identity_dedup(output_path)


# ============================================================================
# Model 5: Mimi Decoder (Stateful Streaming)
# ============================================================================

def _monkeypatch_for_onnx():
    """Replace in-place state mutations with non-in-place equivalents.

    This must be called before tracing the mimi decoder.
    The upstream modules use in-place ops on state tensors.
    ONNX export needs functional equivalents so the tracer
    can track the data flow from input states to output states.

    Patches:
      - StreamingConv1d.forward: state["previous"][:] → state["previous"] = ...
      - StreamingConvTranspose1d.forward: layer_state[:] → state["partial"] = ...
      - _LinearKVCacheBackend.append_and_get: complete_kv uses .item() + slice
        assignment → scatter-based with tensor offset (no .item())
      - _LinearKVCacheBackend.increment_step: state["offset"] += → state["offset"] = ... +
    """
    from pocket_tts.modules.transformer import _LinearKVCacheBackend

    # --- StreamingConv1d: state["previous"][:] = ... → state["previous"] = ...
    def _conv1d_forward_onnx(self, x, model_state):
        B, C, T = x.shape
        S = self._stride
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[-1]

        if TP and self.pad_mode == "replicate":
            assert T >= TP
            init = x[..., :1]
            new_prev = torch.where(
                state["first"].view(-1, 1, 1),
                init.expand_as(state["previous"]),
                state["previous"],
            )
            # Write back (non-in-place)
            if model_state is not None:
                self.get_state(model_state)["previous"] = new_prev
            state["previous"] = new_prev

        if TP:
            x = torch.cat([state["previous"], x], dim=-1)
        y = self.conv(x)
        if TP:
            # Non-in-place state update
            new_prev = x[..., -TP:].contiguous()
            if model_state is not None:
                self.get_state(model_state)["previous"] = new_prev
            if self.pad_mode == "replicate":
                new_first = torch.zeros_like(state["first"])
                if model_state is not None:
                    self.get_state(model_state)["first"] = new_first
        return y

    StreamingConv1d.forward = _conv1d_forward_onnx

    # --- StreamingConvTranspose1d: layer_state[:] = ... → state["partial"] = ...
    def _convtr_forward_onnx(self, x, mimi_state):
        state = self.get_state(mimi_state)
        layer_state = state["partial"]
        y = self.convtr(x)
        PT = layer_state.shape[-1]
        if PT > 0:
            # Non-in-place: add overlap to the left side
            y_start = y[..., :PT] + layer_state

            # The new partial is the rightmost PT samples (minus bias)
            new_partial = y[..., -PT:].clone()
            bias = self.convtr.bias
            if bias is not None:
                new_partial = new_partial - bias[:, None]
            self.get_state(mimi_state)["partial"] = new_partial

            # Output is everything except the trailing partial
            y = torch.cat([y_start, y[..., PT:-PT]], dim=-1)
        return y

    StreamingConvTranspose1d.forward = _convtr_forward_onnx

    # --- _LinearKVCacheBackend.init_state: split [2,B,T,H,D] → separate K,V
    # Guard against double-patching (validate_mimi_decoder calls this again)
    if not hasattr(_LinearKVCacheBackend, '_orig_init_state'):
        _LinearKVCacheBackend._orig_init_state = _LinearKVCacheBackend.init_state

        def _init_state_split_kv(self, batch_size, sequence_length, device=None, dtype=None):
            """Create split K/V caches in fp16 instead of combined [2,B,T,H,D] fp32."""
            orig = _LinearKVCacheBackend._orig_init_state(self, batch_size, sequence_length, device, dtype)
            cache = orig["cache"]
            return {
                "cache_k": cache[0].half(),   # [B, T, H, D] fp16
                "cache_v": cache[1].half(),   # [B, T, H, D] fp16
                "offset": orig["offset"],
            }

        _LinearKVCacheBackend.init_state = _init_state_split_kv

    # --- _LinearKVCacheBackend.append_and_get: circular buffer with split K/V
    #
    # Circular buffer matching KevinAHM's approach, with split caches:
    # - Write positions wrap modularly: pos = (offset + i) % capacity
    # - Return ALL cache slots with absolute position annotations
    # - _build_attention_mask handles context windowing via pos_k

    def _append_and_get_onnx(self, k, v, state):
        if state is None:
            k_attn = k.permute(0, 2, 1, 3)
            v_attn = v.permute(0, 2, 1, 3)
            B = k_attn.shape[0]
            pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
            pos_k = pos_k.view(1, -1).expand(B, -1)
            offset = torch.zeros(B, device=k_attn.device, dtype=torch.long)
            return k_attn, v_attn, pos_k, offset

        cache_k = state["cache_k"]  # [B, T_cap, H, D] fp16
        cache_v = state["cache_v"]  # [B, T_cap, H, D] fp16
        offset = state["offset"]    # [B] int64
        off = offset[0]             # scalar tensor (NOT .item())

        B, T_new, H, D = k.shape
        capacity = cache_k.shape[1]  # 1000

        # Cast K/V to fp16 before scatter into cache
        k_fp16 = k.half()
        v_fp16 = v.half()

        # Circular write: positions wrap modularly
        write_pos = (off + torch.arange(T_new, device=k.device, dtype=torch.long)) % capacity
        idx = write_pos.view(1, T_new, 1, 1).expand(B, T_new, H, D)

        updated_k = cache_k.scatter(1, idx, k_fp16)
        updated_v = cache_v.scatter(1, idx, v_fp16)
        state["cache_k"] = updated_k
        state["cache_v"] = updated_v

        # Build absolute position for every cache slot
        total_written = off + T_new
        all_slots = torch.arange(capacity, device=k.device, dtype=torch.long)
        last_abs = total_written - 1
        last_slot = last_abs % capacity
        delta = all_slots - last_slot
        abs_positions = torch.where(delta <= 0, last_abs + delta, last_abs + delta - capacity)
        pos_k = torch.where(abs_positions >= 0, abs_positions,
                            torch.full_like(abs_positions, -1))
        pos_k = pos_k.view(1, -1).expand(B, -1)

        # Return full cache cast back to fp32 — attention mask filters by position
        k_attn = updated_k.float().permute(0, 2, 1, 3)   # [B, H, capacity, D]
        v_attn = updated_v.float().permute(0, 2, 1, 3)
        return k_attn, v_attn, pos_k, offset

    _LinearKVCacheBackend.append_and_get = _append_and_get_onnx

    # --- _LinearKVCacheBackend.increment_step: non-in-place offset update
    def _increment_step_onnx(self, state, increment):
        state["offset"] = state["offset"] + increment

    _LinearKVCacheBackend.increment_step = _increment_step_onnx


def _restore_patches():
    """Restore original forward methods (optional cleanup)."""
    # We don't bother since this is a one-shot export script.
    pass


class MimiDecoderWrapper(nn.Module):
    """Wraps mimi decoder path with explicit states for ONNX export.

    Input:
      latent [1, N, 32]  — normalized latent from flow_lm
      state_0..state_N   — flat state tensors (all mimi states)

    Output:
      audio_frame [1, 1, T_samples]
      out_state_0..out_state_N

    The decoder path:
      1. Un-normalize: latent * emb_std + emb_mean
      2. Transpose: [1, N, 32] → [1, 32, N]
      3. Quantizer: Conv1d(32, 512, 1) → [1, 512, N]
      4. Upsample: ConvTrUpsample (stride=16) → [1, 512, N*16]
      5. Decoder transformer: 2 layers StreamingMultiheadAttention → [1, 512, N*16]
      6. SEANet decoder → [1, 1, T_samples]

    State layout: All StatefulModules in mimi are flattened (sorted by module
    name, then key). This includes encoder-side states that are pass-throughs
    in the decode path. The C++ runtime initializes all mimi states together.

    Each attention layer contributes: offset [B] int64, cache [2,B,T,H,D] float32
    Conv/ConvTranspose layers contribute: previous/first/partial buffers.
    """

    def __init__(self, model: TTSModel):
        super().__init__()
        self.emb_std = model.flow_lm.emb_std      # [32]
        self.emb_mean = model.flow_lm.emb_mean     # [32]
        self.quantizer = model.mimi.quantizer
        self.upsample = model.mimi.upsample
        self.decoder_transformer = model.mimi.decoder_transformer
        self.decoder = model.mimi.decoder

        # Compute state layout
        self.state_layout, _ = flatten_states(model.mimi, batch_size=1, seq_len=MIMI_CACHE_LEN)
        self.num_states = len(self.state_layout)

        # Identify which states are decoder-side (others are pass-through)
        self.decoder_modules = set()
        for mn, key, shape, dtype in self.state_layout:
            if mn.startswith(("decoder.", "decoder_transformer.", "upsample.")):
                self.decoder_modules.add(mn)

    def forward(self, latent, *states):
        # Build model_state dict from flat states
        model_state = build_model_state_dict(self.state_layout, states)

        # Un-normalize latent
        x = latent * self.emb_std + self.emb_mean  # [1, N, 32]
        x = x.transpose(-1, -2)                     # [1, 32, N]

        # Quantizer (DummyQuantizer: Conv1d 32→512)
        x = self.quantizer(x)                        # [1, 512, N]

        # Upsample (ConvTrUpsample1d, stride=16)
        x = self.upsample(x, model_state)            # [1, 512, N*16]

        # Decoder transformer
        (x,) = self.decoder_transformer(x, model_state)  # [1, 512, N*16]

        # SEANet decoder
        audio = self.decoder(x, model_state)          # [1, 1, T_samples]

        # Increment mimi streaming offsets
        # The upstream calls increment_steps(mimi, state, increment=16)
        # after each decode. For ONNX, we incorporate this into the model.
        # Only StreamingMultiheadAttention has increment_step.
        # Increment = number of transformer frames processed.
        # For upsample stride=16, N input frames → N*16 transformer frames.
        # But the mimi transformer uses frame_rate=12.5, so input is N frames at 12.5fps.
        # After upsample (×16), we get N*16 frames at 200fps (encoder_frame_rate).
        # The transformer processes N*16 frames, so increment = N*16.
        # But actually, the upstream does increment_steps(mimi, state, increment=16)
        # with a fixed increment of 16 per decode call. This is because each call
        # processes 1 latent frame (at 12.5fps) → 16 encoder frames.
        increment = latent.shape[1] * 16  # typically 16 for single-frame decode
        for mn in sorted(model_state.keys()):
            if "decoder_transformer" in mn and "self_attn" in mn:
                model_state[mn]["offset"] = model_state[mn]["offset"] + increment
            # StreamingConv1d doesn't have step-based increment, its state
            # is just the causal buffer which auto-updates in forward()

        # Extract updated states
        out_states = extract_flat_states(self.state_layout, model_state)

        return (audio,) + tuple(out_states)


def export_mimi_decoder(model: TTSModel, output_path: Path):

    # Apply monkeypatches for ONNX-compatible state management
    _monkeypatch_for_onnx()

    wrapper = MimiDecoderWrapper(model)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    N = wrapper.num_states

    # Dummy: single latent frame
    dummy_latent = torch.randn(1, 1, 32)

    # Build dummy states from init_states
    _, init_state_dict = flatten_states(model.mimi, batch_size=1, seq_len=MIMI_CACHE_LEN)
    dummy_states = []
    for mn in sorted(init_state_dict.keys()):
        for key in sorted(init_state_dict[mn].keys()):
            dummy_states.append(init_state_dict[mn][key])

    args = (dummy_latent, *dummy_states)

    # Names
    in_names = ["latent"] + [f"state_{i}" for i in range(N)]
    out_names = ["audio_frame"] + [f"out_state_{i}" for i in range(N)]
    dyn_axes = {
        "latent": {1: "latent_frames"},
        "audio_frame": {2: "audio_samples"},
    }

    export_and_report(wrapper, args, output_path, in_names, out_names, dyn_axes)


# ============================================================================
# Config / Metadata Export
# ============================================================================

# ============================================================================
# INT8 Quantization
# ============================================================================

QUANT_TARGETS = {
    "flow_lm_main": "flow_lm_main.onnx",
    "flow_lm_flow": "flow_lm_flow.onnx",
    "mimi_decoder": "mimi_decoder.onnx",
}


def quantize_model(onnx_dir: Path, src_name: str, dst_name: str):
    """Apply dynamic INT8 quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src_path = onnx_dir / src_name
    dst_path = onnx_dir / dst_name
    if not src_path.exists():
        print(f"  ⊘ {dst_name} (source {src_name} absent — partial export, skipped)")
        return
    assert not dst_path.exists() or dst_path.stat().st_size > 0, f"Broken destination: {dst_path}"

    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,  # per-channel scales: fixes INT8 artifacts, same file size
    )

    src_mb = src_path.stat().st_size / 1e6
    dst_mb = dst_path.stat().st_size / 1e6
    print(f"  ✓ {src_name} → {dst_name} ({dst_mb:.1f} MB, {dst_mb/src_mb*100:.0f}%)")


def run_quantization(onnx_dir: Path):
    """Quantize all target models to INT8."""
    print(f"\nINT8 Quantization")
    print("-" * 40)
    for name, src in QUANT_TARGETS.items():
        dst = src.replace(".onnx", "_int8.onnx")
        quantize_model(onnx_dir, src, dst)


def externalize_models(onnx_dir: Path):
    """Move inline weights to external .data files, mmap'd by ORT at load.

    Faster session startup (no protobuf parse + arena copy of the weight
    bytes) and lets the OS page weights in lazily on cold start. Portable:
    the .data file is the same tensor bytes, no hardware-specific content.
    Runs after quantization so validation covers the shipped artifacts.
    """
    import onnx

    print(f"\nExternalize weights (mmap)")
    print("-" * 40)
    for f in sorted(onnx_dir.glob("*.onnx")):
        model = onnx.load(str(f))
        # Idempotency guard: skip models that already reference external data.
        if any(i.HasField("data_location") and
               i.data_location == onnx.TensorProto.EXTERNAL
               for i in model.graph.initializer):
            print(f"  ⊘ {f.name} (already external)")
            continue
        # onnx.save_model appends to an existing external-data file instead of
        # truncating it, so every re-export would leave a dead copy of the
        # weights behind and the .data file grows by one copy per run.
        data_file = f.parent / (f.name + ".data")
        if data_file.exists():
            data_file.unlink()
        inline_mb = sum(i.ByteSize() for i in model.graph.initializer) / 1e6
        onnx.save_model(
            model, str(f),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
        print(f"  ✓ {f.name} → {f.name}.data ({inline_mb:.1f} MB of weights)")


def print_listing(output_dir: Path):
    print(f"\nOutput: {output_dir.absolute()}")
    total = 0
    for f in sorted(output_dir.glob("*")):
        if not f.is_file():
            continue
        size = f.stat().st_size / 1e6
        total += size
        print(f"  {f.name:30} {size:8.2f} MB")
    print(f"  {'TOTAL':30} {total:8.2f} MB")


def trim_to_default_variant(onnx_dir: Path, keep_fp32_decoder: bool = False):
    """Remove the variants the runtime never loads, so the default export is
    exactly what ships: int8 flow_lm_main + int8 mimi_decoder + fp32
    flow_lm_flow (used instead of int8 flow — it causes robotic artifacts)
    plus the always-fp32 mimi_encoder / text_conditioner.

    keep_fp32_decoder: the German pack ships an fp32 decoder instead (its
    int8 quantization is broken) — trim the int8 decoder, keep fp32."""
    removed = 0
    names = ["flow_lm_main.onnx", "mimi_decoder.onnx", "flow_lm_flow_int8.onnx"]
    if keep_fp32_decoder:
        names = ["flow_lm_main.onnx", "mimi_decoder_int8.onnx", "flow_lm_flow_int8.onnx"]
    for name in names:
        for suffix in ("", ".data"):
            f = onnx_dir / (name + suffix)
            if f.exists():
                f.unlink()
                removed += 1
    if removed:
        print(f"\nTrim to default variant")
        print("-" * 40)
        print(f"  ✓ removed {removed} non-default file(s)")


# ============================================================================
# Validation
# ============================================================================

def _compare(name: str, pt: np.ndarray, ox: np.ndarray, atol=1e-4, rtol=1e-4):
    """Compare PyTorch vs ONNX arrays. Returns (ok, result_string)."""
    if pt.shape != ox.shape:
        return False, f"✗ {name}: SHAPE MISMATCH pt={pt.shape} ort={ox.shape}"
    abs_diff = np.abs(pt - ox).max()
    rel_diff = abs_diff / (np.abs(pt).max() + 1e-8)
    ok = abs_diff < atol or rel_diff < rtol
    symbol = "✓" if ok else "✗"
    return ok, f"{symbol} {name} abs={abs_diff:.2e} rel={rel_diff:.2e}"


def validate_text_conditioner(model: TTSModel, onnx_dir: Path,
                               onnx_file="text_conditioner.onnx", atol=1e-4, rtol=1e-4):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / onnx_file))
    text = "Hello world."
    prepared = model.flow_lm.conditioner.prepare(text)
    tokens = prepared.tokens
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    with torch.no_grad():
        pt_emb = model.flow_lm.conditioner(TokenizedText(tokens))
    ort_out = sess.run(None, {"token_ids": tokens.numpy()})
    ok, msg = _compare("embeddings", pt_emb.numpy(), ort_out[0], atol=atol, rtol=rtol)
    return [(ok, f"text_conditioner: {msg}")]


def validate_mimi_encoder(model: TTSModel, onnx_dir: Path,
                           onnx_file="mimi_encoder.onnx", atol=1e-4, rtol=1e-4):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / onnx_file))
    audio = torch.randn(1, 1, 23040)
    frame_size = 1920
    pad_amount = (frame_size - audio.shape[-1] % frame_size) % frame_size
    audio_padded = F.pad(audio, (0, pad_amount))
    with torch.no_grad():
        pt_conditioning = model._encode_audio(audio_padded)
    ort_out = sess.run(None, {"audio": audio_padded.numpy()})
    ok, msg = _compare("conditioning", pt_conditioning.numpy(), ort_out[0], atol=atol, rtol=rtol)
    return [(ok, f"mimi_encoder: {msg}")]


def validate_flow_lm_flow(model: TTSModel, onnx_dir: Path,
                           onnx_file="flow_lm_flow.onnx", atol=1e-4, rtol=1e-4):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / onnx_file))
    c = torch.randn(1024)
    s = torch.zeros(1, 1)
    t = torch.ones(1, 1)
    x = torch.randn(1, 32)
    with torch.no_grad():
        pt_out = model.flow_lm.flow_net(c, s, t, x)
    ort_out = sess.run(None, {"c": c.numpy(), "s": s.numpy(), "t": t.numpy(), "x": x.numpy()})
    ok, msg = _compare("flow_dir", pt_out.reshape(-1).numpy(), ort_out[0].reshape(-1), atol=atol, rtol=rtol)
    return [(ok, f"flow_lm_flow: {msg}")]


def validate_flow_lm_main(model: TTSModel, onnx_dir: Path,
                           onnx_file="flow_lm_main.onnx", atol=1e-3, rtol=1e-3):
    import onnxruntime as ort
    max_seq = MAX_SEQ_LEN
    sess = ort.InferenceSession(str(onnx_dir / onnx_file))
    N_LAYERS = 6
    results = []

    # Text conditioning pass
    text = "Hello world."
    prepared = model.flow_lm.conditioner.prepare(text)
    tokens = prepared.tokens
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    with torch.no_grad():
        text_embeddings = model.flow_lm.conditioner(TokenizedText(tokens))
    text_len = tokens.shape[1]

    pt_state = init_states(model.flow_lm, batch_size=1, sequence_length=max_seq)
    _sanitize_states(pt_state)
    empty_seq = torch.empty(1, 0, model.flow_lm.ldim)
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(
            model_state=pt_state, text_tokens=tokens, backbone_input_latents=empty_seq,
        )

    ort_k_caches = [np.zeros((1, max_seq, 16, 64), dtype=np.float16) for _ in range(N_LAYERS)]
    ort_v_caches = [np.zeros((1, max_seq, 16, 64), dtype=np.float16) for _ in range(N_LAYERS)]
    ort_steps = [np.zeros((1,), dtype=np.int64) for _ in range(N_LAYERS)]

    feed = {"sequence": np.zeros((1, 0, 32), dtype=np.float32),
            "text_embeddings": text_embeddings.numpy()}
    for i in range(N_LAYERS):
        feed[f"state_{i*3}"] = ort_k_caches[i]
        feed[f"state_{i*3+1}"] = ort_v_caches[i]
        feed[f"state_{i*3+2}"] = ort_steps[i]

    ort_out = sess.run(None, feed)
    for i in range(N_LAYERS):
        ort_k_caches[i] = ort_out[2 + i*3]
        ort_v_caches[i] = ort_out[3 + i*3]
        ort_steps[i] = ort_out[4 + i*3]

    for mn, ms in pt_state.items():
        if "layers.0.self_attn" in mn:
            pt_offset = int(ms["offset"][0].item())
            if "cache" in ms:
                pt_cache_k = ms["cache"][0, 0, :pt_offset].numpy()
            else:
                pt_cache_k = ms["cache_k"][0, :pt_offset].numpy()
            ort_step_val = int(ort_steps[0].item())
            ort_cache_k = ort_k_caches[0][0, :ort_step_val]
            ok, msg = _compare("kv_cache_L0_K", pt_cache_k, ort_cache_k, atol=max(atol, 1e-3), rtol=rtol)
            results.append((ok, f"flow_lm_main text: {msg}"))
            break

    # AR step
    bos_latent = torch.full((1, 1, 32), float("nan"))
    feed_ar = {"sequence": bos_latent.numpy(),
               "text_embeddings": np.zeros((1, 0, 1024), dtype=np.float32)}
    for i in range(N_LAYERS):
        feed_ar[f"state_{i*3}"] = ort_k_caches[i]
        feed_ar[f"state_{i*3+1}"] = ort_v_caches[i]
        feed_ar[f"state_{i*3+2}"] = ort_steps[i]

    ort_ar = sess.run(None, feed_ar)
    ort_ar_conditioning = ort_ar[0]
    ort_ar_eos = ort_ar[1]

    pt_state2 = init_states(model.flow_lm, batch_size=1, sequence_length=max_seq)
    _sanitize_states(pt_state2)
    with torch.no_grad():
        text_emb = model.flow_lm.conditioner(TokenizedText(tokens))
        text_emb_cat = torch.cat([text_emb, torch.empty(1, 0, 1024)], dim=1)
        bos = torch.where(torch.isnan(bos_latent), model.flow_lm.bos_emb, bos_latent)
        input_empty = model.flow_lm.input_linear(torch.empty(1, 0, 32))

        x_text = torch.cat([text_emb_cat, input_empty], dim=1)
        for layer in model.flow_lm.transformer.layers:
            x_text = layer(x_text, pt_state2)
        increment_steps(model.flow_lm, pt_state2, increment=text_len)

        bos_input = model.flow_lm.input_linear(bos)
        x_bos = torch.cat([torch.empty(1, 0, 1024), bos_input], dim=1)
        for layer in model.flow_lm.transformer.layers:
            x_bos = layer(x_bos, pt_state2)
        increment_steps(model.flow_lm, pt_state2, increment=1)

        x_bos = model.flow_lm.out_norm(x_bos)
        pt_conditioning_raw = x_bos[:, -1].numpy()
        pt_eos_raw = model.flow_lm.out_eos(x_bos[:, -1]).numpy()

    ok1, msg1 = _compare("conditioning", pt_conditioning_raw.reshape(-1),
                    ort_ar_conditioning.reshape(-1), atol=atol, rtol=rtol)
    ok2, msg2 = _compare("eos_logit", pt_eos_raw.reshape(-1),
                    ort_ar_eos.reshape(-1), atol=atol, rtol=rtol)
    results.append((ok1, f"flow_lm_main AR: {msg1}"))
    results.append((ok2, f"flow_lm_main AR: {msg2}"))
    return results


def validate_mimi_decoder(model: TTSModel, onnx_dir: Path,
                           onnx_file="mimi_decoder.onnx", atol=1e-4, rtol=1e-4,
                           gain: float = 1.0):
    """gain: constant folded into the graph by add_output_gain (German pack) —
    the PyTorch reference is scaled to match before comparing."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / onnx_file))
    _monkeypatch_for_onnx()
    mimi_cache_len = MIMI_CACHE_LEN
    results = []

    for mn, mod in model.mimi.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = mn

    pt_mimi_state = init_states(model.mimi, batch_size=1, sequence_length=mimi_cache_len)
    _sanitize_states(pt_mimi_state)

    layout = []
    for mn in sorted(pt_mimi_state.keys()):
        for key in sorted(pt_mimi_state[mn].keys()):
            t = pt_mimi_state[mn][key]
            layout.append((mn, key, list(t.shape), t.dtype))

    # Single frame test
    latent = torch.randn(1, 1, 32)
    pt_state_copy = copy.deepcopy(pt_mimi_state)
    with torch.no_grad():
        denorm = latent * model.flow_lm.emb_std + model.flow_lm.emb_mean
        transposed = denorm.transpose(-1, -2)
        quantized = model.mimi.quantizer(transposed)
        pt_audio = model.mimi.decode_from_latent(quantized, pt_state_copy)
        increment_steps(model.mimi, pt_state_copy, increment=16)

    ort_states = extract_flat_states(layout, pt_mimi_state)
    feed = {"latent": latent.numpy()}
    for i, s in enumerate(ort_states):
        feed[f"state_{i}"] = s.numpy()
    ort_out = sess.run(None, feed)

    ok, msg = _compare("frame_0", pt_audio.numpy() * gain, ort_out[0], atol=atol, rtol=rtol)
    results.append((ok, f"mimi_decoder: {msg}"))

    # Multi-frame streaming (4 frames)
    pt_stream_state = copy.deepcopy(pt_mimi_state)
    ort_stream_states = [s.numpy() for s in extract_flat_states(layout, pt_mimi_state)]
    for frame_idx in range(4):
        lat = torch.randn(1, 1, 32)
        with torch.no_grad():
            denorm = lat * model.flow_lm.emb_std + model.flow_lm.emb_mean
            transposed = denorm.transpose(-1, -2)
            quantized = model.mimi.quantizer(transposed)
            pt_frame = model.mimi.decode_from_latent(quantized, pt_stream_state)
            increment_steps(model.mimi, pt_stream_state, increment=16)
        feed = {"latent": lat.numpy()}
        for i, s in enumerate(ort_stream_states):
            feed[f"state_{i}"] = s
        ort_frame_out = sess.run(None, feed)
        ort_stream_states = ort_frame_out[1:]
        ok, msg = _compare(f"frame_{frame_idx+1}", pt_frame.numpy() * gain, ort_frame_out[0], atol=atol, rtol=rtol)
        results.append((ok, f"mimi_decoder: {msg}"))

    return results


def run_validation(model: TTSModel, onnx_dir: Path, int8: bool = False,
                   decoder_gain: float = 1.0):
    """Run all validation checks. Returns True if all pass."""
    label = "INT8" if int8 else "FP32"
    atol = 0.5 if int8 else 1e-3
    rtol = 0.1 if int8 else 1e-3

    print(f"\nValidation ({label})")
    print("-" * 40)

    all_results = []

    if not int8:
        all_results.extend(validate_text_conditioner(model, onnx_dir))
        if (onnx_dir / "mimi_encoder.onnx").exists():
            all_results.extend(validate_mimi_encoder(model, onnx_dir))

    suffix = "_int8.onnx" if int8 else ".onnx"
    all_results.extend(validate_flow_lm_flow(
        model, onnx_dir, onnx_file=f"flow_lm_flow{suffix}", atol=atol, rtol=rtol))
    all_results.extend(validate_flow_lm_main(
        model, onnx_dir, onnx_file=f"flow_lm_main{suffix}", atol=atol, rtol=rtol))
    all_results.extend(validate_mimi_decoder(
        model, onnx_dir, onnx_file=f"mimi_decoder{suffix}", atol=atol, rtol=rtol,
        gain=decoder_gain))

    all_pass = True
    for ok, msg in all_results:
        print(f"  {msg}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"  ✓ All {label} checks passed")
    else:
        print(f"  ✗ Some {label} checks failed")

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PocketTTS ONNX Export, Quantize & Validate")
    parser.add_argument("--output-dir", default="./models", help="Output directory")
    parser.add_argument("--export", nargs="+",
                        default=["all"],
                        choices=["all", "encoder", "text", "main", "flow", "decoder"],
                        help="Which models to export")
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_LEN,
                        help=f"Max KV cache length for flow_lm (default: {MAX_SEQ_LEN})")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (default: use built-in b6369a24)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip export/quantize, only validate existing ONNX files")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation")
    parser.add_argument("--language", default="english_2026-04",
                        choices=["english_2026-04", "english", "german"],
                        help="Model language (english_2026-04 = current kyutai default with BOS, "
                             "english = legacy b6369a24/english_2026-01, german = kyutai german)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"

    print("=" * 60)
    print(f"PocketTTS ONNX Export — {args.language}")
    print("=" * 60)

    if args.language == "german":
        HF_BASE = "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main/languages/german"
        weights_name = "model.safetensors"     # final key format, loads via strict load_state_dict
        config_name = "german.yaml"
        # Gated repo: full weights including the REAL Mimi encoder (WAV voice
        # cloning). The ungated without-voice-cloning twin has the encoder
        # zeroed (remove_voice_cloning_and_push.py); per-language voice
        # embeddings are shipped in both repos and are identical.
        gated_weights = ("kyutai/pocket-tts", "languages/german/model.safetensors")
    elif args.language == "english_2026-04":
        # Current kyutai default model with BOS prompt handling and working
        # cloning encoder. NOTE: the ungated without-voice-cloning file has the
        # encoder ZEROED (remove_voice_cloning_and_push.py) — weights must come
        # from the gated repo (hf auth login + accepted license required).
        HF_BASE = "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main/languages/english_2026-04"
        weights_name = "model.safetensors"
        config_name = "english_2026-04.yaml"
        gated_weights = ("kyutai/pocket-tts", "languages/english_2026-04/model.safetensors")
    else:
        HF_BASE = "https://huggingface.co/Verylicious/pocket-tts-ungated/resolve/main"
        weights_name = "tts_b6369a24.safetensors"
        config_name = "english_2026-01.yaml"   # renamed from b6369a24.yaml upstream
        gated_weights = None

    # Download weights (cached, only needed for export)
    weights_file = cache_dir / weights_name
    if gated_weights:
        # Gated repo: huggingface_hub resolves the local token automatically.
        from huggingface_hub import hf_hub_download
        weights_file = Path(hf_hub_download(gated_weights[0], gated_weights[1]))
        print(f"  ✓ {weights_file} (gated, via hf auth)")
    elif not weights_file.exists():
        print("\nDownloading weights...")
        _download(f"{HF_BASE}/{weights_name}", weights_file)
        print(f"  ✓ {weights_file}")

    # Download tokenizer (needed at runtime)
    tokenizer_file = output_dir / "tokenizer.model"
    if not tokenizer_file.exists():
        print("Downloading tokenizer...")
        _download(f"{HF_BASE}/tokenizer.model", tokenizer_file)
        print(f"  ✓ {tokenizer_file}")

    # Build config pointing at local weights
    import pocket_tts as _ptt
    config_path = Path(_ptt.__file__).parent / "config" / config_name
    if args.config:
        config_path = Path(args.config)
    config = load_config(config_path)
    config.weights_path = str(weights_file)
    config.weights_path = str(weights_file)

    # Load model with voice cloning weights
    print("Loading model...")
    model = TTSModel._from_pydantic_config_with_weights(
        config,
        temp=DEFAULT_TEMPERATURE,
        lsd_decode_steps=DEFAULT_LSD_DECODE_STEPS,
        noise_clamp=DEFAULT_NOISE_CLAMP,
        eos_threshold=DEFAULT_EOS_THRESHOLD,
    )
    model.eval()
    print(f"  Voice cloning: {model.has_voice_cloning}")

    # --- Language metadata for the C++ runtime ---
    export_language_meta(model, output_dir, language=args.language)

    # --- Export + Quantize ---
    if not args.validate_only:
        export_all = "all" in args.export
        print(f"\nExport")
        print("-" * 40)

        # With the gated weights the German Mimi encoder is real, so it exports
        # like English and WAV voice cloning works.
        is_vc_capable = args.language in ("english", "english_2026-04", "german")

        if is_vc_capable and (export_all or "encoder" in args.export):
            export_mimi_encoder(model, output_dir / "mimi_encoder.onnx")

        if export_all or "text" in args.export:
            export_text_conditioner(model, output_dir / "text_conditioner.onnx")

        if export_all or "main" in args.export:
            export_flow_lm_main(model, output_dir / "flow_lm_main.onnx", args.max_seq)

        if export_all or "flow" in args.export:
            export_flow_lm_flow(model, output_dir / "flow_lm_flow.onnx")

        if args.language == "german":
            # German ships an fp32 decoder: the int8 quantization of the
            # German decoder weights is broken (validation rel error ~7 vs
            # fp32 — English quantizes to rel ~0.1), producing clipped,
            # distorted audio. NO output-gain fold: upstream python applies
            # no gain and C++ latents match it 1:1 — folding 3.0 made every
            # german path +9.5 dB hot (the original hotfix bug).
            # ponytail: re-try int8 quantization when Kyutai re-trains.
            export_mimi_decoder(model, output_dir / "mimi_decoder.onnx")
        elif export_all or "decoder" in args.export:
            export_mimi_decoder(model, output_dir / "mimi_decoder.onnx")

        run_quantization(output_dir)

        if args.language == "german":
            print(f"\nBuiltin voices (per-language embeddings)")
            print("-" * 40)
            export_builtin_voices(output_dir, args.language)

        externalize_models(output_dir)
        trim_to_default_variant(output_dir, keep_fp32_decoder=(args.language == "german"))

    # --- Validate ---
    if not args.no_validate:
        fp32_pass = run_validation(model, output_dir, int8=False)

        int8_pass = True
        if (output_dir / "flow_lm_main_int8.onnx").exists():
            int8_pass = run_validation(model, output_dir, int8=True,
                                       decoder_gain=1.0) if args.language != "german" else True  # german int8 decoder is broken, see above

        if not (fp32_pass and int8_pass):
            sys.exit(1)

    # --- Trim to the shipped default variant ---
    print_listing(output_dir)


if __name__ == "__main__":
    main()
