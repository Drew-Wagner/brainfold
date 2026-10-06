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
for converting to and from braindecode `EEGNet`s, `[recipes]` for the training recipes.

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

```python
from packed_eegnet import fit, predict, seeded_packed_eegnet

# X: (n_trials, n_chans, n_times), y: (n_trials,), shared by all runs.
# train_idx: (K, n_train), test_idx: (K, n_test), one row of trial indices per run.
model = seeded_packed_eegnet(seeds, n_chans=22, n_outputs=4, n_times=513)  # member k from seeds[k]
fit(model.cuda(), X, y, train_idx, seeds, epochs=300)
preds = predict(model, X, test_idx)  # (n_test, K)
```

`fit` is a plain Adam + cosine-schedule loop. To write your own, the pieces are:

```python
logits = model(x)                 # x: (batch, K, n_chans, n_times) -> (batch, K, n_outputs)
loss = packed_loss(logits, y)     # y: (batch, K); SUM of member mean losses
loss.backward()                   # so each member gets exactly its own gradient
optimizer.step()                  # one Adam over all parameters == K separate Adams
```

`packed_batches(train_idx, batch_size, generators)` yields `(batch, K)` index
tensors with a separate shuffle per member.

To use braindecode for everything after training (evaluation, saving, the
Hugging Face Hub), convert the pack: `model.to_braindecode()` returns K
ordinary `EEGNet`s, and `PackedEEGNet.from_braindecode(models)` packs existing
ones. `to_state_dicts()` and `load_state_dicts()` do the same with state dicts.
`PackOfOne(eegnet)` gives a single braindecode model the packed interface, so
`fit` and `predict` can train it alone as a baseline.

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
at a time.

| training set | EEGNet | braindecode | K=4 | K=9 | K=18 | K=36 |
|---|---|---|---|---|---|---|
| 4608 trials (leave-one-subject-out) | F1=4, D=2 | 0.110 | 0.068 | 0.028 | **0.017 (6.3x)** | 0.018 |
| | F1=8, D=2 | 0.119 | 0.063 | 0.029 | **0.028 (4.3x)** | 0.029 |
| 288 trials (leave-one-session-out) | F1=4, D=2 | 0.0080 | 0.0041 | 0.0019 | **0.0011 (7.6x)** | 0.0011 |
| | F1=8, D=2 | 0.0075 | 0.0042 | 0.0019 | **0.0017 (4.4x)** | 0.0018 |

Throughput stops improving at about K=18. With K=1, `PackedEEGNet` is about
2x *slower* than braindecode's `EEGNet`; the gain appears from about K=4.

**Why packing alone isn't enough:** EEGNet's first block (temporal conv over every
electrode at full time resolution, BatchNorm, spatial conv) does real work, and
that work grows with K. Only the later, tiny layers are launch-bound and pack
for free. Most of the speedup comes from reordering the first block.

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
