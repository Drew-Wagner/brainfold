"""Numerical error of the recipes' speed options, against float64, on a recipe's own data.

    python benchmarks/approximation.py recipes/moabb/hparams/bnci2014001_eegnet.yaml
    python benchmarks/approximation.py recipes/moabb/hparams/bnci2014001_atcnet.yaml --epochs 20 --out a.csv

The first pack of the recipe (seed 0) is trained with dropout off, so the only
differences between options are numerical, from the same initial weights:

- one step: the training logits and the gradients on one batch, as
  ||option - float64|| / ||float64||, over all gradients and for the worst
  parameter. Parameters whose float64 gradient is below 1e-6 of the total are
  left out of the worst: the first BatchNorm's gamma and beta feed a conv and a
  second BatchNorm, which cancels them up to eps, so their gradients are ~0;
- after --epochs epochs of the recipe's training loop: the test logits' relative
  error and the fraction of test trials whose predicted class differs from
  float64's (training amplifies rounding, so this bounds what reaches results).

float32 recomputes the first BatchNorm's input moments every step, as before
input_moments; every other option precomputes them. Each option's error is
next to float32's, the error of plain float32 training.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "moabb"))
from train import ARCHITECTURES, fit, folds, load_cached, load_hparams, packs, predict  # noqa: E402

# name: (precompute moments, compile, autocast dtype, TF32 matmuls, data dtype)
OPTIONS = {
    "float32": (False, False, None, False, torch.float32),
    "float32, moments": (True, False, None, False, torch.float32),
    "compile": (True, True, None, False, torch.float32),
    "tf32": (True, False, None, True, torch.float32),
    "bf16": (True, False, torch.bfloat16, True, torch.float32),
    "float16 data": (True, False, None, False, torch.float16),
    "compile, bf16, float16 data": (True, True, torch.bfloat16, True, torch.float16),
}


class Recompute(torch.nn.Module):
    """Hides ``input_moments``, so ``fit`` recomputes the moments every step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def source_optimizer_param_groups(self, *args):
        return self.model.source_optimizer_param_groups(*args)


def no_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p = 0.0
    return model


def relative(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def one_step(model, x, y, moments, compile, autocast_dtype):
    """Training logits and gradients of the summed member losses on one batch."""
    model = copy.deepcopy(model).train()
    forward = torch.compile(model) if compile else model
    with torch.autocast(x.device.type, autocast_dtype, enabled=autocast_dtype is not None):
        logits = forward(x, moments) if moments is not None else forward(x)
    F.cross_entropy(logits.float().permute(0, 2, 1), y, reduction="none").mean(0).sum().backward()
    return logits.float(), {name: p.grad for name, p in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs before comparing test logits")
    parser.add_argument("--pack-size", type=int, help="default: the recipe's")
    parser.add_argument("--options", nargs="+", default=list(OPTIONS), choices=list(OPTIONS))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "brainfold")
    parser.add_argument("--out", type=Path, help="write one CSV row per option")
    args = parser.parse_args()
    hp = load_hparams(args.hparams, [])
    data, train, evaluation = hp["data"], hp["train"], hp["evaluation"]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    X, y, subjects, sessions = load_cached(data, args.cache_dir)
    n_classes = len(set(y.tolist()))
    runs = [(name, 0, torch.from_numpy(tr), torch.from_numpy(te))
            for name, tr, te in folds(evaluation["protocol"], y, subjects, sessions, evaluation.get("n_folds"))]
    pack = next(packs(runs, args.pack_size or train["pack_size"]))
    K, seeds = len(pack), [0] * len(pack)
    train_idx = torch.stack([r[2] for r in pack]).to(device)
    test_idx = torch.stack([r[3] for r in pack]).to(device)
    X32, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    base = no_dropout(ARCHITECTURES[hp["architecture"]].from_seeds(
        list(range(K)), n_chans=X.shape[1], n_outputs=n_classes, n_times=X.shape[2], **hp["model"])).to(device)
    settings = dict(epochs=args.epochs, lr=train["lr"], batch_size=train["batch_size"],
                    weight_decay=train.get("weight_decay"), class_weight=train.get("class_weight"))
    print(f"{hp['architecture']} on {data['dataset']}: {K} runs, dropout off, {args.epochs} epochs", flush=True)

    # float64 reference; TF32 is off for it (float64 never uses it, but be explicit)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    batch = train_idx[:, : train["batch_size"]].T  # (batch, K)
    ref_logits, ref_grads = one_step(base.double(), X32[batch].double(), y[batch], None, False, None)
    ref_model, _ = fit(Recompute(copy.deepcopy(base).double()), X32.double(), y, train_idx, seeds, **settings)
    ref_test = predict(ref_model, X32.double(), test_idx)

    rows = []
    for name in args.options:
        precompute, compile, autocast_dtype, tf32, data_dtype = OPTIONS[name]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = True  # PyTorch's default: cuDNN convs may use TF32
        torch._dynamo.reset()
        model = copy.deepcopy(base).float()
        Xd = X32.to(data_dtype)
        x = Xd[batch].float()
        moments = None
        if precompute:
            moments = tuple(m.view(*batch.shape, *m.shape[1:]) for m in model.input_moments(x.flatten(0, 1)))
        logits, grads = one_step(model, x, y[batch], moments, compile, autocast_dtype)
        total = torch.cat([g.flatten() for g in ref_grads.values()]).norm()
        grad_error = relative(torch.cat([grads[n].flatten() for n in grads]),
                              torch.cat([ref_grads[n].flatten() for n in grads]))
        worst, worst_name = max((relative(grads[n], ref_grads[n]), n) for n in grads
                                if ref_grads[n].norm() > 1e-6 * total)
        trained, _ = fit(model if precompute else Recompute(model), Xd, y, train_idx, seeds,
                         compile=compile, autocast_dtype=autocast_dtype, **settings)
        test = predict(trained, Xd, test_idx, autocast_dtype=autocast_dtype)
        flips = (test.argmax(-1) != ref_test.argmax(-1)).double().mean().item()
        rows.append(dict(option=name, step_logits=relative(logits, ref_logits), step_grads=grad_error,
                         step_grads_worst=worst, step_grads_worst_parameter=worst_name,
                         test_logits=relative(test, ref_test), prediction_flips=flips))
        r = rows[-1]
        print(f"{name:30s} one step: logits {r['step_logits']:.1e}, gradients {r['step_grads']:.1e} "
              f"(worst {r['step_grads_worst']:.1e}, {worst_name});"
              f"  after {args.epochs} epochs: test logits {r['test_logits']:.1e}, "
              f"predictions differing {100 * flips:.2f}%", flush=True)
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
