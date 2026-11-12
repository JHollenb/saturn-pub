"""Pretrained last-digit consumer; native prefixes, frozen original KL anchors."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from protocol import digest, plain


def cases(seed, count=8):
    rng = np.random.default_rng(seed)
    result = []
    for digit in rng.permutation(10)[:count]:
        a = int(rng.integers(10, 90))
        b = int(rng.integers(1, 8))*10 + ((int(digit)-a) % 10)
        result.append({"prompt": "Write only the last digit of the sum.\n17 + 25 = 2\n36 + 48 = 4\n" + str(a) + " + " + str(b) + " = ",
                       "answer": str((a+b) % 10)})
    return result


PRESERVE = ["The capital of France is", "The opposite of hot is", "A week has", "Water freezes at",
            "The capital of Japan is", "The opposite of up is", "A triangle has", "The color of snow is"]
FRESH_PRESERVE = ["The capital of Italy is", "The opposite of left is", "A year has", "Ice melts at",
                  "The capital of Spain is", "The opposite of slow is", "A square has", "The color of grass is"]


class Consumer(torch.nn.Module):
    def __init__(self, net, tokenizer, seed, *, evaluation=False, examples=None, preservation=None):
        super().__init__()
        self.net = net
        self.examples = cases(seed + (100003 if evaluation else 0)) if examples is None else examples
        self.preservation_prompts = (FRESH_PRESERVE if evaluation else PRESERVE) if preservation is None else preservation
        self.task_count = len(self.examples)
        self.preservation_count = len(self.preservation_prompts)
        inputs = tokenizer([r["prompt"] for r in self.examples] + self.preservation_prompts,
                           padding=True, return_tensors="pt", add_special_tokens=False)
        if inputs.input_ids.shape[1] > 128:
            raise ValueError("native input length exceeds declared geometry")
        device = next(net.parameters()).device
        self.register_buffer("input_ids", inputs.input_ids.to(device))
        self.register_buffer("attention_mask", inputs.attention_mask.to(device))
        targets = [tokenizer.encode(r["answer"], add_special_tokens=False) for r in self.examples]
        if any(len(t) != 1 for t in targets):
            raise ValueError("answer must be a single native token")
        self.register_buffer("target_ids", torch.tensor([t[0] for t in targets] + [0]*self.preservation_count, device=device))
        self.register_buffer("is_task", torch.tensor([True]*self.task_count + [False]*self.preservation_count, device=device))
        with torch.inference_mode():
            baseline = net(input_ids=self.input_ids, attention_mask=self.attention_mask,
                use_cache=False, logits_to_keep=1).logits[:, -1, :].float().log_softmax(-1)
        self.register_buffer("anchor_logp", baseline.clone())
        self.last_readout = None
        self.input_identity = digest({"ids": plain(self.input_ids.cpu()), "mask": plain(self.attention_mask.cpu()),
            "targets": plain(self.target_ids.cpu()), "anchor_bytes_sha256": hashlib.sha256(baseline.cpu().numpy().tobytes()).hexdigest()})
        self.eval().requires_grad_(False)

    def comparison_parameter_identity(self):
        from mdb.wsp_weights import parameter_inventory
        return parameter_inventory(self)["identity"]

    def comparison_input_identity(self):
        return digest({name: {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()}
            for name in ("input_ids", "attention_mask", "target_ids", "is_task", "anchor_logp")
            for value in (getattr(self, name),)})

    def forward(self, rows):
        if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("native rows must be a one-dimensional integer tensor")
        if len(rows) == 0 or len(torch.unique(rows)) != len(rows) or bool((rows < 0).any()) or bool((rows >= len(self.input_ids)).any()):
            raise ValueError("native rows must be unique valid input indices")
        rows = rows.to(device=self.input_ids.device, dtype=torch.long)
        self.last_readout = None
        logits = self.net(input_ids=self.input_ids[rows], attention_mask=self.attention_mask[rows],
                          use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("native logits contain nonfinite values")
        logp = logits.log_softmax(-1)
        correct = logp.gather(1, self.target_ids[rows, None]).squeeze(1)
        anchor = self.anchor_logp[rows]
        kl = (anchor.exp() * (anchor-logp)).sum(-1)
        loss = torch.where(self.is_task[rows], -correct, kl)
        self.last_readout = {"rows": rows.detach().cpu().numpy().copy(),
            "logp": logp.detach().cpu().numpy().copy(), "correct_logp": correct.detach().cpu().numpy().copy(),
            "kl": kl.detach().cpu().numpy().copy(), "winner_ids": logits.argmax(-1).detach().cpu().numpy().copy()}
        # Only this inspection representation clips roundoff-negative KL. The
        # authoritative scorer below retains the signed native KL reductions.
        return loss.clamp_min(0).sqrt().unsqueeze(-1)

    def comparison_score(self, splits, names):
        rows = torch.cat([splits[n][0] for n in names])
        if not torch.equal(rows.cpu(), torch.arange(len(self.input_ids))):
            raise ValueError("native scorer requires the fixed ordered task/preservation table")
        with torch.inference_mode():
            self(rows)
        return self.readout_score()

    def readout_score(self):
        r = self.last_readout
        target_logp = r["correct_logp"][:self.task_count]
        kl = r["kl"][self.task_count:]
        return {"task_mse": float(-target_logp.mean()), "preservation_mse": float(kl.mean()),
            "objective": float(target_logp.mean()-kl.mean()), "task_mean_logp": float(target_logp.mean()),
            "preservation_kl": float(kl.mean()), "score_units": "NLL and KL nats; compatibility names task_mse/preservation_mse are not MSE in this profile",
            "task_outputs": (-target_logp).tolist(), "preservation_outputs": kl.tolist(),
            "target_logp": target_logp.tolist(), "winner_ids": r["winner_ids"][:self.task_count].tolist(),
            "target_ids": self.target_ids[:self.task_count].cpu().tolist(),
            "correct": int(np.count_nonzero(r["winner_ids"][:self.task_count] == self.target_ids[:self.task_count].cpu().numpy()))}

    def anchor_score(self):
        logp = self.anchor_logp.cpu().numpy()
        targets = self.target_ids.cpu().numpy()
        self.last_readout = {"rows": np.arange(len(logp)), "logp": logp.copy(),
            "correct_logp": logp[np.arange(len(logp)), targets].copy(), "kl": np.zeros(len(logp), dtype=logp.dtype),
            "winner_ids": logp.argmax(-1)}
        return self.readout_score()


def load(config, root):
    from qwen_family_campaign import prepare_parameters
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    checkpoint = Path(config["model_path"])
    h = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("pretrained checkpoint differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    net = Qwen2ForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").to(config["device"]).eval().requires_grad_(False)
    prepare_parameters(checkpoint, net, root)
    return net, tokenizer


def context(net, tokenizer, seed, *, evaluation=False):
    model = Consumer(net, tokenizer, seed, evaluation=evaluation)
    sites = ["net.model.layers." + str(i) + ".mlp.down_proj" for i in range(net.config.num_hidden_layers)]
    splits = {"discovery": (torch.arange(8), torch.zeros(8, 1)),
              "preservation": (torch.arange(8, 16), torch.zeros(8, 1))}
    return {"model": model, "splits": splits, "sites": sites,
        "input_identity": {"native_table": model.input_identity}, "guard_model": net, "virtual_gain": True}


@contextmanager
def gains(model, edits):
    modules = dict(model.named_modules())
    handles = []
    try:
        for edit in edits:
            gain = edit["gain"]
            handles.append(modules[edit["site"]].register_forward_hook(lambda module, args, out, gain=gain: out*gain))
        yield
    finally:
        for h in handles:
            h.remove()


def save_readout(model, path, *, compact=False):
    arrays = dict(model.last_readout)
    if compact:
        logp = arrays.pop("logp")
        arrays["logp_sha256"] = np.frombuffer(hashlib.sha256(logp.tobytes()).digest(), dtype=np.uint8)
        arrays["logp_shape"] = np.asarray(logp.shape, dtype=np.int64)
    np.savez_compressed(path, **arrays)
    return {"path": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def packet(ctx):
    model = ctx["model"]
    parameters = dict(model.named_parameters())
    return {"model_identity": model.comparison_parameter_identity(), "input_identity": ctx["input_identity"],
        "architecture": "original Qwen2.5-0.5B-Instruct, CPU FP32 eager; fixed full-prefix batches, no KV",
        "sites": ctx["sites"], "weight_summaries": {s: {"shape": list(parameters[s+".weight"].shape),
            "rms": float(parameters[s+".weight"].square().mean().sqrt())} for s in ctx["sites"]},
        "examples": model.examples, "preservation_prompts": model.preservation_prompts,
        "baseline": model.comparison_score(ctx["splits"], ("discovery", "preservation"))}
