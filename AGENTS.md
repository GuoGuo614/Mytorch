# KernelLeaf maintenance rules

- `kernelleaf/` is the only canonical framework. The previous Needle framework
  was removed after V0 and must not be reintroduced or mixed into KernelLeaf.
- Preserve the small `TensorOp.compute` / `TensorOp.gradient` automatic
  differentiation architecture. Do not replace it with grad-function closures.
- Work through `KernelLeaf_refactor.md` in order and implement only
  the version explicitly requested by the user. Current repository structure
  takes precedence where the historical assumptions in that document conflict.
- NumPy CPU must always work. CuPy, Triton, and NCCL are optional and must be
  lazily imported. Never silently mix devices or copy GPU arrays to CPU in an
  operator hot path.
- Do not use PyTorch as a runtime dependency or tensor/autograd implementation.
- Every optimization needs a correctness fallback, tests, and a benchmark.
- Do not add datasets, checkpoints, build products, caches, videos, or large
  benchmark logs to Git.
- Preserve license/source attribution for adapted code.
- Use the Conda environment `donkey-env2` for project tests in this workspace.
- At the end of each migration version, report changes, exact tests, unverified
  hardware paths, diff stats, and decisions needed for the next version.
- Never commit or push migration work unless the user explicitly requests it.
