# packed-eegnet

Train many EEGNets at once, 4-7x faster per model on a GPU.

EEG benchmarks train hundreds of small EEGNets (seeds x folds x subjects), one
at a time. `PackedEEGNet` trains K of them as one network and computes
EEGNet's first block in a cheaper order. Each member computes exactly what a
separate braindecode `EEGNet` (Lawhern et al., 2018) would: the tests check
forward passes, BatchNorm running statistics and Adam training steps against K
separate models in float64.

## Install

```bash
pip install "packed-eegnet @ git+https://github.com/Drew-Wagner/packed-eegnet"
```

The package depends only on PyTorch. Extras: `[braindecode]` (braindecode >= 1.2)
for `to_braindecode()` and `from_braindecode()`, `[recipes]` for the recipes.

For development:

```bash
uv sync --group dev --extra braindecode --extra recipes
uv run pytest
```

## Usage

`PackedEEGNet` takes braindecode `EEGNet`'s arguments, with the same names,
defaults and meaning, after the pack size: `PackedEEGNet(K, **kwargs)` is K
`EEGNet(**kwargs)`. That includes `chs_info`, `sfreq` and
`input_window_seconds` for inferring `n_chans` and `n_times`.
`PackedEEGNet.from_seeds(seeds, ...)` initializes member k from `seeds[k]`
alone, so a member's initial weights don't depend on what it is packed with.

```python
import torch
import torch.nn.functional as F
from packed_eegnet import PackedEEGNet

model = PackedEEGNet.from_seeds(seeds, n_chans=22, n_outputs=4, n_times=512).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in batches:  # x: (batch, K, n_chans, n_times), y: (batch, K); column k is member k's batch
    logits = model(x)  # (batch, K, n_outputs)
    losses = F.cross_entropy(logits.permute(0, 2, 1), y, reduction="none").mean(0)  # (K,)
    optimizer.zero_grad()
    losses.sum().backward()  # SUM, not mean: each member gets exactly its own gradient
    optimizer.step()         # one Adam over the packed parameters == K separate Adams
```

Not every training trick keeps the members independent: see
[Keeping members independent](#keeping-members-independent). The recipe below
has a complete loop: per-member shuffling, a cosine schedule, evaluation and a
baseline.

**Per-member state.** `member_state_dict(k)` returns member k's weights and
BatchNorm statistics, and `load_member_state_dict(k, state)` overwrites member k
alone. For example, per-member early stopping:

```python
for k in range(K):
    if val_acc[k] > best_acc[k]:
        best_acc[k], best_state[k] = val_acc[k], model.member_state_dict(k)
...
model.load_state_dicts(best_state)  # every member at its own best epoch
```

**braindecode.** The state dicts use braindecode `EEGNet`'s format.
`model.to_braindecode()` returns K ordinary `EEGNet`s for everything after
training (evaluation, saving, the Hugging Face Hub), and
`PackedEEGNet.from_braindecode(models)` packs existing ones.

## Recipes

`recipes/bnci2014001_loso/` trains EEGNet on BNCI2014001 (BCI Competition IV
2a, 4-class motor imagery), leave-one-session-out: 9 subjects x 2 folds = 18
runs per seed, all trained as one pack. Data loading and preprocessing use
braindecode (`MOABBDataset`, 4-38 Hz band-pass, resampling to 128 Hz,
exponential moving standardization, `create_windows_from_events`).
Hyperparameters are in `hparams.yaml`; override them with `--set`:

```bash
cd recipes/bnci2014001_loso
python train.py                                  # 300 epochs, seed 0
python train.py --set train.seeds=[0,1,2]
python train.py --baseline       # also train every run alone as a braindecode EEGNet
python train.py --set data.subjects=[1] train.epochs=3 --device cpu   # smoke test
```

`--baseline` trains every run a second time as a separate braindecode
`EEGNet`, from the same initial weights and with the same data order. With
seeds 0, 1, 2 (54 runs) on an RTX 4080 SUPER, training and evaluation:

| | accuracy | kappa | time |
|---|---|---|---|
| PackedEEGNet, packs of 18 | 0.669 ± 0.165 | 0.559 ± 0.221 | 29 s |
| braindecode EEGNet, one at a time | 0.670 ± 0.162 | 0.560 ± 0.216 | 131 s |

(± is the standard deviation over runs.) The per-run accuracy difference is
-0.001 ± 0.038: the two differ only in their dropout masks.

## Speed

Seconds of training per model per epoch, batch 64, on an RTX 4080 SUPER
(`benchmarks/bench_packing.py`). The baseline trains one braindecode `EEGNet`
at a time; "torch.func" is PyTorch's ensembling recipe over 18 braindecode
`EEGNet`s (see below).

| training set | EEGNet | braindecode | torch.func, K=18 | packed, K=4 | K=9 | K=18 | K=36 |
|---|---|---|---|---|---|---|---|
| 4608 trials (leave-one-subject-out) | F1=4, D=2 | 0.111 | 0.091 (1.2x) | 0.062 | 0.028 | **0.017 (6.7x)** | 0.017 |
| | F1=8, D=2 | 0.115 | 0.190 (0.6x) | 0.062 | 0.028 | **0.027 (4.3x)** | 0.028 |
| 288 trials (leave-one-session-out) | F1=4, D=2 | 0.0075 | 0.0057 (1.3x) | 0.0044 | 0.0020 | **0.0011 (7.1x)** | 0.0011 |
| | F1=8, D=2 | 0.0076 | 0.0119 (0.6x) | 0.0044 | 0.0020 | **0.0017 (4.5x)** | 0.0017 |

Throughput stops improving at about K=18. With K=1, `PackedEEGNet` is about
2x *slower* than braindecode's `EEGNet`; the gain appears from about K=4.

## Why not PyTorch's ensembling recipe?

The standard way to train K copies of a model at once is `torch.func`:
`stack_module_state` on K models, then `vmap` over `functional_call`. On
braindecode's `EEGNet` it is exact: forward passes and BatchNorm running
statistics match K separate models in float64. It just isn't fast here (table
above): 1.2x with F1=4, and *slower* than one model at a time with F1=8.

- **It batches the computation as written.** EEGNet's first block filters every
  electrode at full time resolution before mixing electrodes, and that work
  grows with K. `vmap` can only run it K times in parallel. `PackedEEGNet`
  mixes electrodes first (below), which does about 11x less work in that block.
- **Batched convolutions become grouped convolutions.** On CUDA these run in
  PyTorch's generic depthwise kernel (`conv_depthwise2d`), not cuDNN. In a
  profile of one K=18 training step, those kernels take more than half of the
  GPU time.
- braindecode's max-norm constraint uses `renorm`, which has no `vmap`
  batching rule and falls back to a loop over members (PyTorch warns about
  it). Replacing it with an equivalent batchable expression did not change
  the timing measurably, so it isn't the bottleneck.

Most of `PackedEEGNet`'s speedup comes from the reordered first block. Packing
the later layers, which are small and launch-bound, into grouped convolutions
does the rest.

## How it works

**Packing.** The K members sit side by side on the channel axis. Every conv is
grouped so that member k only sees its own channels. BatchNorm, dropout and
max-norm act per channel or per output row anyway.

**Reordered first block.** The temporal conv `w_f` (length L, F1 filters), its
BatchNorm (gamma, beta) and the depthwise spatial conv `s_g` (D per temporal
filter, over C electrodes) are all linear in the input x. For spatial filter g
reading temporal filter f:

```
z_g = a_f * (w_f * (s_g . x)) + (beta_f - a_f * mu_f) * sum(s_g),   a_f = gamma_f / sqrt(var_f + eps)
```

So the C electrodes are mixed into F1*D signals first (one matmul), and the
length-L filter then runs on F1*D signals instead of F1*C electrode channels:
11x less for C=22, D=2. The (F1, C, T) tensor is never built.

In training, BatchNorm needs the batch mean and variance of `u_f = w_f * x` over
(batch, electrodes, time). Both are quadratic in `w_f`:

```
mu_f = w_f . e,    E[u_f^2] = w_f^T M w_f,    var_f = E[u_f^2] - mu_f^2
```

Here `e` (length L) and `M` (L x L) are the window means and lag second moments
of the zero-padded input. `M` comes from the FFT autocorrelation of the input,
minus the products each window leaves out at the two edges, read off small
Gram matrices of the edge samples. Neither needs gradients. Gradients reach
w, gamma, beta and s through `mu_f` and `var_f` exactly as through the original
BatchNorm, and the running statistics are updated the same way.

## Keeping members independent

A packed model is one `nn.Module` whose parameters hold all K members side by
side. Each member trains exactly as it would alone only if **every operation
in the training loop is separable across members**: it acts on each parameter
element on its own, or it is a sum of per-member terms. Anything that computes
one statistic over the whole pack (a norm, a maximum, a scale, a threshold, a
shape) couples the members, and one member's training then depends on what it
is packed with.

Separable, so safe:

- The summed loss `losses.sum()`. A *mean* over members also keeps them
  separate, but scales every gradient by 1/K, which changes SGD's effective
  learning rate and Adam's balance against `eps`.
- Elementwise optimizers: SGD (with momentum), Adam, AdamW, RMSprop. Weight
  decay, gradient accumulation, EMA and SWA weight averaging.
- Learning-rate schedules that depend only on the step (cosine, step, warmup).
- Data-parallel training (DDP averages gradients elementwise) and `torch.compile`.

Couples the members:

- **Global gradient clipping:** `clip_grad_norm_(model.parameters(), ...)` uses
  one norm for the whole pack. A member with large gradients shrinks everyone
  else's step. Clip each member's slice separately, or not at all.
- **Mixed-precision loss scaling:** `torch.amp.GradScaler` keeps one scale and
  skips the *whole* optimizer step when any gradient is inf or NaN, so one
  member's overflow skips a step for all of them. bf16 autocast needs no scaler
  and is fine.
- **Optimizers with per-tensor or cross-element statistics:** layer-wise trust
  ratios (LARS, LAMB), factored or matrix preconditioners (Adafactor, Shampoo,
  Muon), and SAM's global perturbation norm all look across a packed tensor or
  the whole model. Check that an optimizer's state is per element before using it.
- **Metric-driven schedules on an aggregate:** `ReduceLROnPlateau` or early
  stopping on the pack's mean validation loss ties every member to the others.
  Track metrics per member, and keep each member's best weights with
  `member_state_dict(k)`.
- **Re-initializing parameters** with `torch.nn.init` on the packed tensors:
  initializers that use the fan-out or the whole matrix (`xavier_*`,
  `orthogonal_`) see the packed shape, not one member's. Use `from_seeds` or
  `load_state_dicts` instead.
- **Regularizers or pruning over the whole model** that aren't sums of
  per-member terms, such as a penalty on the total weight norm (not squared)
  or a global magnitude-pruning threshold.

Shared by construction:

- **Hyperparameters.** All members share the architecture, optimizer settings,
  learning-rate schedule and batch size. Pack runs that differ only in seed
  and data (folds, subjects), not runs from a hyperparameter sweep.
- **Dropout's RNG** (see Caveats).

## Caveats

- **Dropout shares one RNG across the pack.** A member's initial weights and
  data order depend only on its own seed, but its dropout masks depend on which
  runs it is packed with. Results are reproducible for a fixed pack
  composition, not per run.
- **Packed runs need equal training-set sizes:** every step draws one batch per
  member. Subsample to equal sizes or group runs by size.
- **Float32 variance:** `var = E[u^2] - mu^2` can lose precision when the
  temporal-filter output has a large mean relative to its spread. On roughly
  standardized inputs it matches braindecode to about 1e-4 in float32 (tested),
  and the recipe's exponential moving standardization keeps inputs in that
  range. Be careful with raw, un-centered inputs.
- **Unsupported `EEGNet` options** raise a `ValueError`: cropped decoding (a
  `final_conv_length` shorter than the remaining time) and activations with
  parameters, such as `PReLU`, which would be shared across the pack.
  `batch_norm_momentum` must be a float (not `None`).
- **`batch_norm_affine=False`** works in `PackedEEGNet`, but braindecode (as of
  1.8.1) fails to build such an `EEGNet`, so `to_braindecode()` does too.

## License

MIT
