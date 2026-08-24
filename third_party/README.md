# Third-party code: LearningFractals (Tu et al., AAAI 2023)

The real-image comparison (MNIST / Fashion-MNIST, Section 6.7 of the paper) runs
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
