# Third-party code and data

No third-party code or data is redistributed in this repository. The scripts and
commands below fetch them into `refs/` (git-ignored); each keeps its own license.

## LearningFractals (Tu et al., AAAI 2023)

The real-image comparison (MNIST / Fashion-MNIST, Section 6.8 of the revised manuscript) runs
the public implementation of

> Cheng-Hao Tu, Hong-You Chen, David Carlyn, Wei-Lun Chao.
> "Learning Fractals by Gradient Descent." AAAI 2023.
> https://github.com/andytu28/LearningFractals (BSD-3-Clause)

We do not redistribute their code. Fetch it into `refs/LearningFractals`
(the default path expected by our scripts):

```bash
mkdir -p refs
git clone https://github.com/andytu28/LearningFractals refs/LearningFractals
```

Notes:

- Our driver `scripts/run_tu_mnist_balanced10.py` executes their optimization
  unmodified. Their code imports `cv2` and `numba`, which are used only on code
  paths we do not exercise; `scripts/tu_cv2_stub/` provides no-op stubs that the
  driver puts on `PYTHONPATH` so these packages are not required.
- MNIST / Fashion-MNIST are downloaded automatically via `torchvision` into
  `refs/LearningFractals/data` (the layout their code also uses).
- The common-condition evaluation (`scripts/evaluate_mnist_common_rendering.py`)
  follows their evaluation protocol (32x32 RBF + clamp rendering, fixed range
  [-5,5] as in their `evaluate_mse.py`, minimum MSE over 100 sampling
  sequences) and was verified against their own `evaluate_mse.py` to agree to
  within 6e-5 in image space.

## Chaotic fractals (Djeacoumar et al., Eurographics 2025)

The external-benchmark and official-code experiments of the revision
(`scripts/rev1/run_external.sh`, `scripts/rev1/run_i7a2.sh`) use the benchmark and
the public implementation of

> Adarsh Djeacoumar, Felix Mujkanovic, Hans-Peter Seidel, Thomas Leimkühler.
> "Learning Image Fractals Using Chaotic Differentiable Point Splatting."
> Computer Graphics Forum (Eurographics) 2025.
> https://github.com/sshadr/chaotic-fractals and https://chaotic-fractals.mpi-inf.mpg.de

```bash
bash third_party/fetch_chaotic_fractals.sh   # code (pinned commit) -> refs/chaotic-fractals,
                                             # benchmark data        -> refs/chaotic_data
bash third_party/setup_chaotic_env.sh        # separate conda env "chaotic-fractals" (CUDA 12.4)
```

Licenses: their code is under the MIT License (Copyright (c) 2025 Adarsh
Djeacoumar); the Gaussian rasterizer bundled with it
(`lib/diff-isogaussian-rasterization`, `lib/simple-knn`) is under the
Gaussian-Splatting License of Inria and MPII, which permits non-commercial research
use only; for the benchmark data, see their project page.

Notes:

- The fetch script checks out commit `0e47513` and adds one line
  (`#include <cstdint>`) to `cuda_rasterizer/rasterizer_impl.h` so that the rasterizer
  builds with GCC 13 or later; their code is otherwise unmodified.
- Their code runs only in its own env. `scripts/rev1/export_chaotic_codes.py`
  (exports their published results, which are pickled modules, to JSON) and
  `scripts/rev1/i7a2_run_official.py` (draws targets with their ground-truth generator
  and runs their optimizer with its default configuration) are our scripts that call
  their modules; `run_external.sh` and `run_i7a2.sh` invoke them through
  `conda run -n chaotic-fractals`.
- `scripts/rev1/optimize_hybrid.py` is our re-implementation of their
  simulated-annealing schedule in our objective; it does not use their code.
- Their renderer normalizes every attractor by its bounding box and draws maps with
  probabilities proportional to `|det W| + 0.01`; `scripts/rev1/prepare_external_ifs.py`
  and `scripts/rev1/i7a2_score.py` follow these conventions when comparing.
