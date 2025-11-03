# GENERATIVE — Autoencoder, VAE, and DDPM-Mini (v0.1)

This guide adds a gentle, **hands-on** path through three foundational generative families:

- **Autoencoder (AE)** — reconstruct inputs; learn compressed codes.
- **Variational Autoencoder (VAE)** — probabilistic latent variable model; **ELBO** training with **KL warmup**.
- **DDPM-Mini** — tiny diffusion model (28×28) with a very small UNet and just a few timesteps.

Everything runs quickly on **Apple Silicon MPS** or CPU and comes with tiny tests.

> **Math rendering on GitHub:** inline formulas use `$…$` and display equations use `$$…$$`.

---

## Contents
1. [Environment](#environment)
2. [Autoencoder (AE)](#autoencoder-ae)
3. [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
4. [DDPM-Mini (diffusion)](#ddpmmini-diffusion)
5. [Acceptance Gates & Tests](#acceptance-gates--tests)
6. [Learn-By-Tweaking](#learn-by-tweaking)
7. [File Map](#file-map)

---

## Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pytest -q
````

---

## Autoencoder (AE)

**Paths**

* Model: `src/airoad/generative/ae.py`
* Script: `scripts/train_ae_mnist.py`
* Test: `tests/test_ae_vae_ddpm.py::test_ae_train_step_reduces_mse`

### Intuition

An AE learns an **encoder** $f_\phi: x \mapsto z$ and a **decoder** $g_\theta: z \mapsto \hat{x}$, trained to minimize reconstruction error. The latent $z$ is a compact representation of the input.

### Loss (MSE)

```math
\mathcal{L}_\text{AE} = \frac{1}{N}\sum_{i=1}^{N}\big\lVert \hat{x}_i - x_i \big\rVert_2^2
```

We use a small **conv encoder/decoder** with stride-2 downsamples to reach a 64-dim latent, then mirror back with ConvTranspose. Inputs/outputs are in $[0,1]$ (Sigmoid on the last layer).

### Run

```bash
uv run python scripts/train_ae_mnist.py --steps 500 --latent-dim 64 --limit-train 10000
```

---

## Variational Autoencoder (VAE)

**Paths**

* Model: `src/airoad/generative/vae.py`
* Script: `scripts/train_vae_mnist.py`
* Test: `tests/test_ae_vae_ddpm.py::test_vae_elbo_step`

### Intuition

A VAE posits a latent variable $z$ with prior $p(z)=\mathcal{N}(0,I)$ and learns an approximate posterior $q_\phi(z\mid x)=\mathcal{N}!\big(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))\big)$. Samples are drawn via the **reparameterization trick**:

```math
z \;=\; \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I).
```

### ELBO Objective

```math
\mathcal{L}_\text{ELBO}
\;=\;
\underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}
\;-\;
\underbrace{\mathrm{KL}\!\big(q_\phi(z\mid x)\;\|\;p(z)\big)}_{\text{regularization}}.
```

For a diagonal Gaussian $q_\phi(z\mid x)$ and standard normal prior:

```math
\mathrm{KL}\!\big(q_\phi(z\mid x)\;\|\;p(z)\big)
=\frac12 \sum_{j=1}^{d} \left(
\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1
\right).
```

We train with a **Bernoulli decoder** (BCE recon) and a **KL warmup** schedule:

```math
\beta_t \;=\; \min\!\Big(\beta_{\max}, \;\frac{t}{\text{warmup\_steps}}\,\beta_{\max}\Big),
\qquad
\mathcal{L}_t \;=\; \text{BCE} \;+\; \beta_t \,\mathrm{KL}.
```

### Run

```bash
uv run python scripts/train_vae_mnist.py --steps 600 --warmup-steps 200 --latent-dim 32 --limit-train 10000
```

---

## DDPM-Mini (diffusion)

**Paths**

* Model: `src/airoad/generative/ddpm_mini.py`
* Script: `scripts/train_ddpm_mini.py`
* Test: `tests/test_ae_vae_ddpm.py::test_ddpm_loss_and_shapes`

### Intuition

Diffusion models learn to **denoise** data by reversing a simple **forward noising** process. We use a tiny UNet and **very few steps** ($T\in[4,8]$) for speed.

### Forward (noising)

With a variance schedule ${\beta_t}_{t=1}^T$, define $\alpha_t=1-\beta_t$, $\bar{\alpha}*t=\prod*{s=1}^{t}\alpha_s$. Then

```math
x_t \;=\; \sqrt{\bar{\alpha}_t}\,x_0 \;+\; \sqrt{1-\bar{\alpha}_t}\,\epsilon, 
\qquad \epsilon \sim \mathcal{N}(0, I).
```

### Training objective (noise prediction)

The model $\epsilon_\theta(x_t, t)$ predicts $\epsilon$ with MSE:

```math
\mathcal{L}_\text{DDPM} 
\;=\; \mathbb{E}_{t,\epsilon}\big[\,\lVert \epsilon_\theta(x_t, t) - \epsilon \rVert_2^2 \big].
```

### Reverse (denoising) step (DDPM form)

```math
x_{t-1}
\;=\; \frac{1}{\sqrt{\alpha_t}}
\left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,t) \right)
\;+\; \sigma_t z,
\quad z \sim \mathcal{N}(0,I),
\quad \sigma_t=\sqrt{\beta_t}.
```

We keep the UNet **very small** and set $T{=}6$ by default. After training, we sample a tiny grid and save it.

### Run

```bash
uv run python scripts/train_ddpm_mini.py --steps 800 --T 6 --batch-size 128 --limit-train 10000
# outputs a grid to: outputs/ddpm_grid.png
```

---

## Acceptance Gates & Tests

* **AE**: single train step **reduces MSE** on a random batch.
  Test: `test_ae_vae_ddpm.py::test_ae_train_step_reduces_mse`
* **VAE**: ELBO with warmup — later step **≤** initial step (non-increasing).
  Test: `test_ae_vae_ddpm.py::test_vae_elbo_step`
* **DDPM-Mini**: loss is finite; backward works; shape sanity.
  Test: `test_ae_vae_ddpm.py::test_ddpm_loss_and_shapes`

Run all:

```bash
uv run pytest -q
```

---

## Learn-By-Tweaking

* **AE**

  * Change `latent_dim` (e.g., 16→128); watch recon MSE and sample sharpness.
  * Try **BCE** recon instead of **MSE** (`nn.BCELoss`) when inputs are in $[0,1]$.

* **VAE**

  * Sweep `warmup_steps` and `beta_max`; see KL’s effect on blurry vs sharp recon.
  * Visualize the latent by projecting $\mu(x)$ with PCA/TSNE to see clustering.

* **DDPM**

  * Vary $T$ (4/6/8). Fewer steps = faster but harder denoising.
  * Try linear vs cosine-like beta schedules; observe grid quality.

---

## File Map

* `src/airoad/generative/ae.py` — conv **Autoencoder** (MSE recon).
* `src/airoad/generative/vae.py` — conv **VAE** + **ELBO** + warmup helper.
* `src/airoad/generative/ddpm_mini.py` — tiny **UNet**, scheduler, loss, sampler.
* `scripts/train_ae_mnist.py` — AE trainer (MNIST).
* `scripts/train_vae_mnist.py` — VAE trainer (MNIST, KL warmup).
* `scripts/train_ddpm_mini.py` — DDPM trainer + sampling grid.
* `tests/test_ae_vae_ddpm.py` — tiny smoke tests for all three.

---

**Tip:** Save a few sample grids across training runs — you’ll quickly build intuition for how AE/VAEs tend to **reconstruct** while diffusion models **sample** diverse digits from noise.
