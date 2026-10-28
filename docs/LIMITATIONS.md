# Limitations and unverifiable items

- No official MQAdapter code. Circuit and adapter are paper-faithful **reimplementations**.
- CRZ control/target on the ring is unspecified; default `control_first`.
- Layer index “6-th to final”: default 0-indexed blocks 6–11 (6 blocks) to match 0.078M under bias-free projections.
- MaPLe, PromptSRC and MMRL++ are re-implemented from their papers (mqa/hosts/); none was compared with official code.
- Text prompts for MRI are an implementation choice.
- Four classes make Top-K retrieval a weak analogue of ImageNet-scale coarse filtering.
- Dataset statistics must come from `inspect_dataset`; they are not copied from Kaggle cards.
- This environment: no CUDA in-process, no `open_clip`, no TorchQuantum, no MRI files, no CLIP weights. Full CLIP tests and GPU training were **not** run here.
- TorchQuantum is optional; training uses a statevector VQC with the same gates as Eqs. 5–8.
- No experimental accuracy numbers are reported unless a training/eval script actually produced them.

## Added with the completion pass
- The authoring sandbox had **no PyTorch, no GPU and no network**. Only the NumPy oracle tests, the protocol-logic tests and static checks (compile, import consistency) were executed. All torch code, including MaPLe and the TorchQuantum backend, is unexecuted and may contain bugs.
- MaPLe is re-implemented from the published description (n_ctx=2, depth 9, text->vision projections, deep prompt replacement); it was not compared with the official repository. It runs fp32 here (official uses fp16) and uses MRI-friendly augmentation (small rotation, no flip).
- PromptSRC and MMRL++ hosts: see mqa/hosts/ and docs/REPRODUCTION_STATUS.md for inferred details.
- Base-to-new class halves are an arbitrary choice for a 4-class dataset.
- The cross-attention ablation is my reading of 'conventional cross-attention' (single head, 8-d, learned q/k/v); the paper does not give its exact form.
- In `mqa/models/maple.py` the retrieval stage uses the unprompted frozen CLIP image embedding and fixed hand-written class prompts, which is my reading of 'frozen pre-trained CLIP' in Sec. IV-A.
