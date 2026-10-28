# Paper vs implementation compliance

| Paper item | Implemented? | Notes |
|---|---|---|
| CLIP ViT-B/16 | Yes (code path) | `open_clip` OpenAI weights; FakeCLIP for tests without download |
| Frozen CLIP Top-K anchors | Yes | `retrieve_topk` |
| K=5 | No (invalid) | C=4; K∈{2,3,4} |
| Shared 8-qubit VQC Eqs. 5–8 | Yes | Statevector `PaperVQC` + NumPy oracle |
| TorchQuantum | Yes (new, unexecuted) | `mqa/quantum/tq_vqc.py` + `tests/test_tq_vqc.py` (skips without the package) |
| Ring CRZ + Ry + Pauli-Z | Yes | |
| Eq. 11 projections, bias-free | Yes | Matches 77,848 params for 6 layers |
| Eqs. 12–14 attention | Yes | softmax over K, scale √N |
| Eq. 15 residual γ | Yes | default 0.001 |
| Insert from layer 6 to last | Yes | default 6–11 0-indexed (D2) |
| Eq. 16 cosine CE over all C | Yes | CLIP logit_scale |
| Plug-in to MaPLe | **Yes (new)** | `mqa/models/maple.py`; re-implemented from the MaPLe paper/repo, not checked against official code |
| Plug-in to PromptSRC / MMRL++ | **Yes (engine)** | `mqa/hosts/promptsrc.py`, `mqa/hosts/mmrl.py`; re-implemented from their papers |
| Base-to-new | **Yes (new)** | `protocol: base2new`, base={glioma, meningioma}, new={notumor, pituitary} (arbitrary halves) |
| Cross-dataset / domain generalisation | **Yes (engine)** | MRI targets (Figshare, Br35H) + synthetic MRI domain shifts; any folder dataset |
| Few-shot 1/2/4/8/16 | Yes | `shots` key |
| Table V ablations (cross-attention, separate VQC) | Yes (new) | `interaction`, `vqc_sharing`; exact baseline details of the paper's cross-attention are unspecified |
| Fig. 1 / Fig. 5 / Table VI / Table VII analyses | Yes (new) | scripts listed in README |
| 0.078M extra params | Counted at runtime | 77848 under D2/D5 on 768/512 CLIP |
| γ / K / depth ablations | `scripts/run_sweep.py` | Not run in the authoring session |
