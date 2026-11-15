| Model | Donor-moves, native (entry 10) / replica | Readable stratum: cell repair | band repair | Right-operand stratum: cell | band | Excluded (no correct same-country item) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 17/48 / 15/28 | 5/23 | 5/23 | 5/9 | 5/9 | 6 |
| Qwen3-1.7B-Base | 30/48 / 29/41 | 12/17 | 12/17 | 6/10 | 7/10 | 5 |
| Qwen3-4B-Base | 15/48 / 16/42 | 2/11 | 2/11 | 3/8 | 3/8 | 5 |

By depth (readable / right-operand, cell arm):

| Model | d3 | d4 | d5 |
|---|---|---|---|
| Qwen3-0.6B-Base | 1/4 / 1/2 | 2/7 / 0/2 | 2/12 / 4/5 |
| Qwen3-1.7B-Base | 1/2 / 3/4 | 5/5 / 2/3 | 6/10 / 1/3 |
| Qwen3-4B-Base | – / 2/3 | 2/3 / 1/2 | 0/8 / 0/3 |

| Model | Arm 1 move: language / capital | Direct share 1.0: language / capital | MLP share: language / capital | Re-read: language / capital | MLP dose ratio: language / capital | Arm 3 margin ratio: language / capital | Arm 4 complement names: language / capital | Arm 4 complement margin move: language / capital | Any 2x language layer (arm 2) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 20/40 / 15/28 | 0.46 / 0.36 | 0.45 / 0.62 | 0.07 / 0.03 | 2.60 / 2.67 | 0.85 / 0.83 | 32/56 / 28/48 | 0.64 / 0.47 | yes: o26 |
| Qwen3-1.7B-Base | 24/41 / 29/41 | 0.46 / 0.40 | 0.39 / 0.57 | 0.16 / 0.01 | 2.45 / 3.13 | 1.04 / 1.00 | 43/56 / 40/48 | 0.39 / 0.59 | no |
| Qwen3-4B-Base | 25/47 / 16/42 | 0.19 / 0.17 | 0.38 / 0.53 | 0.46 / 0.34 | 2.40 / 2.94 | 0.90 / 0.90 | 46/56 / 46/48 | 0.25 / 0.23 | no |

Per-layer MLP term of the arm-1 ladder (median share of the native delta), language / capital:

- Qwen3-0.6B-Base: L20 +0.05/+0.10, L21 +0.22/+0.15, L22 +0.26/+0.23, L23 +0.16/+0.21, L24 +0.02/+0.06, L25 +0.02/+0.00, L26 -0.03/-0.08, L27 -0.32/-0.12
- Qwen3-1.7B-Base: L20 +0.04/+0.06, L21 +0.20/+0.23, L22 +0.32/+0.24, L23 +0.14/+0.16, L24 +0.08/+0.13, L25 -0.01/-0.02, L26 -0.20/-0.17, L27 -0.16/-0.16
- Qwen3-4B-Base: L24 +0.02/+0.03, L25 +0.03/+0.02, L26 +0.01/+0.03, L27 +0.05/+0.03, L28 +0.08/+0.09, L29 +0.24/+0.22, L30 +0.15/+0.10, L31 +0.08/+0.10, L32 +0.01/+0.07, L33 -0.06/-0.02, L34 -0.01/-0.03, L35 -0.21/-0.11
