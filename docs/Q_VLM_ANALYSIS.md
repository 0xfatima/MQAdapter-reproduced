# Does the paper contain a "Q-VLM" (hybrid quantum VLM) architecture?

Short answer: **MQAdapter *is* the hybrid quantum-classical model. There is no second, separate Q-VLM architecture in the paper.**

What the paper (arXiv:2607.12418) actually contains:

| Piece | Classical or quantum | Where |
|---|---|---|
| CLIP ViT-B/16 image + text encoders (frozen) | classical | Sec. IV, V-A2 |
| Prompt-learning hosts MaPLe / PromptSRC / MMRL++ | classical | Sec. V-A2 |
| Coarse stage: cosine Top-K retrieval of class embeddings (semantic anchors) | classical | Sec. IV-A |
| Linear projections 768->8 (per layer), 512->8 (shared), 8->768 (per layer) | classical | Eq. 11, 15 |
| 8-qubit VQC: Ry angle encoding, Ry, ring CRZ, Ry, Pauli-Z readout (24 params) | **quantum circuit, simulated** | Eqs. 5-8 |
| Softmax cross-modal attention over the K anchors | classical (on VQC outputs) | Eqs. 13-14 |
| Residual injection into the class token, gamma = 0.001 | classical | Eq. 15 |
| Cosine cross-entropy | classical | Eq. 16 |

So the "hybrid" is: classical frozen VLM + classical projections -> shared VQC -> classical attention -> VQC -> classical projection.
The circuit is simulated with TorchQuantum on a GPU (efficiency is reported on an RTX 3090). The text I read does not mention running on quantum
hardware, and it does not describe a fully quantum encoder or a quantum-native VLM.

The related-work section names other quantum / multimodal systems (QViLa, MEDQUA, Quantum-PEFT, QTHA, QPA, QuanTA). They are cited for contrast and are **not** part of this paper's method; the
paper provides no code for them, so they are not reproduced here.

Consequences for this repository: the "Q-VLM" pipeline = CLIP (+ MaPLe host) + MQAdapter plug-in. Both live in `mqa/models/` (`MQAdapterCLIP`, `MaPLeCLIP`, `MQAdapterPlugin`).
PromptSRC and MMRL++ are the two paper hosts that are still **not** implemented (see `docs/COMPLIANCE.md`).
