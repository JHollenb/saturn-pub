"""Execution index: the input-invariant name of a dynamic instruction instance.

An instance is named by (call string, iteration count of every enclosing loop, pc) (advisor memo
§1.3, Xin/Sumner/Zhang PLDI 2008). Loops are natural loops of the static CFG, headed by a node that
dominates its back edge's source (``Binary.loops``), not merely targets of backward jumps.

Rule per tick, in this order:
1. leave loops: pop the top loop while it belongs to the current frame and pc is outside its body;
2. arrive at a loop head: the top loop is this loop in this frame -> iteration + 1, else push it at 0;
3. the key is a 64-bit mix of (pc, call-string hash, the loop stack's (loop, iteration) pairs);
4. after the key: ``call`` pushes the call site into the call string; ``ret`` pops it and drops the
   callee's loops.

Two implementations, one rule: ``readable`` (Python, one lane, tuples for display) and ``keys``
(torch, B lanes at once, used for alignment). ``key_of`` hashes a readable index exactly as
``keys`` does; the tests check they agree.
"""
from __future__ import annotations

import torch

# Two independent polynomial hashes modulo 31-bit primes, packed as h1 << 31 | h2: no intermediate
# exceeds 2^62, so the torch version never relies on signed-overflow wraparound.
P1, P2 = 2147483647, 2147483629
A1, A2 = 1103515245, 69069
SEED, SEED_L = (12345, 54321), (271828, 314159)
MAX_LOOPS, MAX_CALLS = 8, 32


def mix(h: tuple, x: int) -> tuple:
    return ((h[0] * A1 + x % P1) % P1, (h[1] * A2 + x % P2) % P2)


def pack(h: tuple) -> int:
    return (h[0] << 31) | h[1]


def readable(binary, pcs) -> list[tuple]:
    """Per tick: (call sites, ((loop head pc, iteration), ...), pc)."""
    depth, calls, stack, out = 0, [], [], []
    for pc in pcs:
        pc = int(pc)
        r = binary.row_of(pc)
        while stack and stack[-1][2] == depth and not binary.inbody[stack[-1][0], r]:
            stack.pop()
        h = int(binary.head_row[r])
        if h >= 0:
            if stack and stack[-1][0] == h and stack[-1][2] == depth:
                stack[-1][1] += 1
            else:
                stack.append([h, 0, depth])
        out.append((tuple(calls), tuple((binary.loops[s[0]].head, s[1]) for s in stack), pc))
        k = binary.kind[r]
        if k == "call":
            calls.append(pc)
            depth += 1
        elif k == "ret" and calls:
            calls.pop()
            depth -= 1
            while stack and stack[-1][2] > depth:
                stack.pop()
    return out


def key_of(binary, index: tuple) -> int:
    calls, loops, pc = index
    ch = SEED
    for c in calls:
        ch = mix(ch, c)
    head_id = {lp.head: lp.id for lp in binary.loops}
    lh = SEED_L
    for head, it in loops:
        lh = mix(mix(lh, head_id[head]), it)
    return pack(mix(mix(mix(SEED, pc), pack(ch)), pack(lh)))


def text(index: tuple) -> str:
    """``0x401030@0x401005/L0x401010:3`` (pc @ call sites / loop head:iteration)."""
    calls, loops, pc = index
    parts = [f"{c:#x}" for c in calls] + [f"L{h:#x}:{i}" for h, i in loops]
    return f"{pc:#x}" + ("@" + "/".join(parts) if parts else "")


def _tmix(h, x):
    x = x.remainder(P1), x.remainder(P2)
    return (h[0] * A1 + x[0]).remainder(P1), (h[1] * A2 + x[1]).remainder(P2)


def _tpack(h):
    return (h[0] << 31) | h[1]


def _full(B, v, dev):
    return (torch.full((B,), v[0], dtype=torch.int64, device=dev), torch.full((B,), v[1], dtype=torch.int64, device=dev))


def _where(c, a, b):
    return torch.where(c, a[0], b[0]), torch.where(c, a[1], b[1])


def keys(binary, pcs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Execution-index keys int64 [B, T] for pc rows [B, T] (``valid`` False -> key -1)."""
    dev = pcs.device
    B, T = pcs.shape
    sorted_pcs = torch.tensor(binary.pcs, dtype=torch.int64, device=dev)
    kind = torch.tensor([1 if k == "call" else 2 if k == "ret" else 0 for k in binary.kind], device=dev)
    inbody = torch.tensor(binary.inbody, device=dev)
    head = torch.tensor(binary.head_row, device=dev)
    L = MAX_LOOPS
    lid = torch.zeros(B, L, dtype=torch.int64, device=dev)
    it = torch.zeros(B, L, dtype=torch.int64, device=dev)
    ld = torch.zeros(B, L, dtype=torch.int64, device=dev)
    lsp = torch.zeros(B, dtype=torch.int64, device=dev)
    depth = torch.zeros(B, dtype=torch.int64, device=dev)
    ch = _full(B, SEED, dev)
    chs = torch.zeros(B, MAX_CALLS, dtype=torch.int64, device=dev)       # packed call-string hashes
    out = torch.full((B, T), -1, dtype=torch.int64, device=dev)
    ar = torch.arange(L, device=dev)
    overflow = torch.zeros(B, dtype=torch.bool, device=dev)

    def top():
        i = (lsp - 1).clamp(min=0)[:, None]
        return i, lid.gather(1, i)[:, 0], ld.gather(1, i)[:, 0]

    for t in range(T):
        pc, v = pcs[:, t].contiguous(), valid[:, t]
        j = torch.searchsorted(sorted_pcs, pc).clamp(max=len(binary.pcs) - 1)
        r = torch.where(sorted_pcs[j] == pc, j + 1, 0)
        for _ in range(L):
            i, tl, td = top()
            pop = v & (lsp > 0) & (td == depth) & ~inbody[tl, r]
            if not bool(pop.any()):
                break
            lsp = lsp - pop.long()
        h = head[r]
        i, tl, td = top()
        same = (lsp > 0) & (tl == h) & (td == depth)
        inc = v & (h >= 0) & same
        push = v & (h >= 0) & ~same
        it = it + (inc[:, None] & (ar == i)).long()
        overflow |= push & (lsp >= L)
        at = (ar == lsp.clamp(max=L - 1)[:, None]) & push[:, None]
        lid = torch.where(at, h[:, None], lid)
        it = torch.where(at, 0, it)
        ld = torch.where(at, depth[:, None], ld)
        lsp = (lsp + push.long()).clamp(max=L)
        lh = _full(B, SEED_L, dev)
        for q in range(L):
            if not bool((lsp > q).any()):
                break
            lh = _where(lsp > q, _tmix(_tmix(lh, lid[:, q]), it[:, q]), lh)
        key = _tpack(_tmix(_tmix(_tmix(_full(B, SEED, dev), pc), _tpack(ch)), _tpack(lh)))
        out[:, t] = torch.where(v, key, -1)
        k = kind[r]
        call, ret = v & (k == 1), v & (k == 2) & (depth > 0)
        overflow |= call & (depth >= MAX_CALLS - 1)
        slot = depth.clamp(max=MAX_CALLS - 1)[:, None]
        chs = chs.scatter(1, slot, torch.where(call, _tpack(ch), chs.gather(1, slot)[:, 0])[:, None])
        ch = _where(call, _tmix(ch, pc), ch)
        depth = depth + call.long() - ret.long()
        back = chs.gather(1, depth.clamp(max=MAX_CALLS - 1)[:, None])[:, 0]
        ch = _where(ret, (back >> 31, back & P1), ch)
        if bool(ret.any()):
            for _ in range(L):
                i, tl, td = top()
                pop = ret & (lsp > 0) & (td > depth)
                if not bool(pop.any()):
                    break
                lsp = lsp - pop.long()
    if bool(overflow.any()):
        raise ValueError(f"loop nesting deeper than {MAX_LOOPS} or calls deeper than {MAX_CALLS} "
                         f"on {int(overflow.sum())} lanes")
    return out


def instances(pcs) -> list[int]:
    """Per tick: k such that this tick is the k-th (0-based) execution of its pc."""
    seen: dict[int, int] = {}
    out = []
    for pc in pcs:
        pc = int(pc)
        out.append(seen.get(pc, 0))
        seen[pc] = out[-1] + 1
    return out
