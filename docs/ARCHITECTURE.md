# Architecture (paper components ↔ this code)

MQAdapter is a **coarse-to-fine** adapter on CLIP ViT-B/16 (Zhao et al., arXiv:2607.12418).

## Coarse stage (Section IV-A)

Frozen CLIP image embedding vs all C class text embeddings → Top-K class indices. Those text embeddings are semantic anchors \(P\in\mathbb{R}^{K\times d_t}\).

Code: `MQAdapterCLIP.retrieve_topk` (`mqa/models/mqadapter_clip.py`).

On this MRI task, **C = 4**. K=5 is invalid. Configs provide K=2, K=3, and K=4. **K=4 retrieves every class** (control).

## Fine stage (Section IV-C, Eqs. 11–15)

Inserted on visual **class tokens** after transformer blocks **6–11** (0-indexed; implementation choice D2 matching ~0.078M extra params).

For each adapted layer \(\ell\):

1. \(\bar c_\ell = W^v_\ell c_\ell\) (768→8, bias-free)
2. \(\bar P = W^t P\) (512→8, **shared** across layers)
3. \(\tilde c_\ell = f_\phi(\bar c_\ell)\), \(\tilde P = f_\phi(\bar P)\) with **one shared VQC**
4. \(\alpha=\mathrm{softmax}(\tilde P^\top \tilde c / \sqrt{N})\), \(h=\tilde P\alpha\)
5. \(\hat c_\ell = c_\ell + \gamma W_\ell f_\phi(h)\), default \(\gamma=0.001\)

Code: `MQAdapterLayer` + `MQAdapterCLIP.encode_image_adapted`.

## VQC (Section IV-B, Eqs. 5–8)

N=8 qubits, 24 parameters:

- Angle encoding \(R_y(x_n)\) on \(\lvert 0\rangle\)
- \(R_y(\phi^{(1)})\)
- Ring CRZ, \(E=\{(n,n+1)\}\cup\{(N,1)\}\)
- \(R_y(\phi^{(2)})\)
- Pauli-Z expectation on each qubit

Training implementation: batched **statevector** in `mqa/quantum/torch_vqc.py` (`PaperVQC`).  
Oracle: `mqa/quantum/reference_paper_vqc.py` (NumPy).  
Optional TorchQuantum sketch: `scripts/phase3_env_check.py` (not required for training).

This is **not** an MLP standing in for a quantum circuit. Ablations that *do* replace \(f_\phi\) are named (`quantum_map`: `mlp`, `identity`, `cos_bias`, `frozen_vqc`).

## Loss (Eq. 16)

Temperature-scaled cosine softmax over **all 4** classes, \(\tau = 1/\texttt{logit\_scale}\) from CLIP (frozen). Retrieval never drops a test image.

## Stand-alone vs paper hosts

The paper plugs MQAdapter into MaPLe / PromptSRC / MMRL++. Those hosts are **not** reimplemented. This repo trains the adapter on **frozen** CLIP (visual + text). That is a documented deviation: Eq. 16 is used, but text prompts are fixed MRI templates, not learned tokens.

## MRI preprocessing (implementation choice D11)

Grayscale→RGB, pad-to-square, bicubic resize 224, CLIP mean/std. No horizontal flip.
