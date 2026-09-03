"""Train the same tiny MLP on CPU and CUDA and compare loss descent."""

import argparse
import numpy as np

import mytorch as mt
import mytorch.nn as nn


def _model(device):
    return nn.Sequential(
        nn.Linear(2, 8, device=device),
        nn.ReLU(),
        nn.Linear(8, 2, device=device),
    )


def train_on_device(device, initial_state, steps=30):
    inputs = mt.Tensor(
        np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], dtype=np.float32),
        device=device,
        requires_grad=False,
    )
    labels = mt.Tensor(
        np.array([0, 0, 1, 1], dtype=np.int32),
        device=device,
        requires_grad=False,
    )
    model = _model(device)
    model.load_state_dict(initial_state)
    optimizer = mt.optim.SGD(model.parameters(), lr=0.08)
    loss_fn = nn.SoftmaxLoss()
    losses = []
    for _ in range(steps):
        optimizer.reset_grad()
        loss = loss_fn(model(inputs), labels)
        losses.append(float(loss.numpy()))
        loss.backward()
        optimizer.step()
    return losses


def run(include_cuda=True, steps=30):
    np.random.seed(7)
    initial_state = _model(mt.cpu()).state_dict()
    results = {"cpu": train_on_device(mt.cpu(), initial_state, steps)}
    if include_cuda and mt.is_cuda_available():
        results["cuda:0"] = train_on_device(mt.cuda(0), initial_state, steps)
    for name, losses in results.items():
        if losses[-1] >= losses[0]:
            raise RuntimeError(f"{name} loss did not decrease: {losses[0]} -> {losses[-1]}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    results = run(include_cuda=not args.cpu_only, steps=args.steps)
    for name, losses in results.items():
        print(f"{name}: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
