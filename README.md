# FAP — Foveated Adversarial Purification

Reference implementation for the FAP pipeline:
- Gradient-guided **fixations**
- **Eccentricity-adaptive** separable Gaussian **blur**
- **CAM-based lesion masks**
- **Lesion-aware masked PGD** training
- Clean, white-box robustness, and ℓ2 **certification**

## Install
```bash
pip install -e .
