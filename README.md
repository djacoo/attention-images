# Which Parts of Attention Does an Image Actually Need?

Ablation study of the Vision Transformer on CIFAR-10.

Deep Learning project — Machine Learning & Deep Learning, University of Verona, A.Y. 2025-26.

## Question

The Transformer encoder was designed for sequences of words. The Vision Transformer keeps that
encoder unchanged and swaps the tokens: image patches instead of words. Which parts of that
machinery still earn their place after the swap, and which are inherited from language and no longer
paying for themselves?

## Study

One ViT is assembled entirely from the course lab's own encoder blocks — the same
`MultiHeadAttention`, `PositionalEncoding`, `FeedForward`, `LayerNorm`, `SublayerConnection` and
`Encoder` written in Lab 05 — plus patch embedding and a linear head. Reference configuration: 4x4
patches, 6 layers, 4 heads, d_model 192, CLS pooling, 1.79M parameters.

That model is then taken apart one component at a time: positional encoding removed or replaced,
patch size doubled, head count varied from 1 to 8, depth cut to 2 and 4, CLS token replaced by mean
pooling, augmentation switched off. Eleven configurations, three seeds each, an identical epoch
budget and one learning rate chosen once on validation and then frozen. An MLP, a CNN and zero-shot
CLIP provide context. Finally, individual attention heads are ablated at inference to measure the
per-class cost of each one.

## Answers

All figures are test accuracy on CIFAR-10, mean over 3 seeds.

| | accuracy | vs reference |
|---|---|---|
| MLP (Lab 02) | 57.19 ± 0.07 | |
| CNN with dropout (Lab 02) | 74.87 ± 0.87 | |
| CLIP ViT-B/32 zero-shot (Lab 10) | 93.66 | |
| **Reference ViT** | **72.80 ± 0.27** | — |
| no positional encoding | 59.05 ± 0.40 | **−13.75** |
| depth 2 | 65.84 ± 0.40 | −6.95 |
| patch size 8 | 67.20 ± 0.40 | −5.59 |
| augmentation disabled | 67.55 ± 0.69 | −5.24 |
| learned positional embeddings | 68.49 ± 0.84 | −4.31 |
| 1 attention head | 69.82 ± 0.27 | −2.98 |
| depth 4 | 71.09 ± 0.43 | −1.71 |
| 2 attention heads | 71.57 ± 0.68 | −1.22 |
| 8 attention heads | 73.09 ± 0.15 | +0.29 |
| mean pooling instead of CLS | 74.57 ± 0.46 | +1.78 |

The ablations separate into two groups.

**What the image needs is spatial.** Removing positional encoding costs 13.75 points, by far the
largest effect measured, and lands the model at 59.05 — barely above the 57.19 of an MLP that never
sees spatial structure at all. Without position, the encoder degenerates into a bag of patches.
Halving the depth costs 6.95, doubling the patch size costs 5.59, and removing augmentation costs
5.24.

**What the image does not need is inherited from language.** The CLS token, taken from BERT-style
text encoders, is worse than simply averaging the patch tokens: mean pooling gains 1.78 points and
produces the best configuration in the study at 74.57, statistically level with the CNN's 74.87.
Learned positional embeddings, strictly more expressive than a fixed sinusoid, lose 4.31 points —
45k images are not enough to learn what the sinusoid encodes for free. And attention capacity
saturates early: 1 to 2 to 4 heads gains 1.75 then 1.23 points, while 4 to 8 gains 0.29, inside the
seed noise.

Head ablation confirms that the heads are not interchangeable. Removing a single head costs 1.03
points on average, but the effect is concentrated: the most important head (layer 1, head 3) costs
2.40 points averaged over classes and 9.50 points on `horse` alone, while the least important costs
0.27. A roughly ninefold spread in importance across heads of the same model.

## Reading the results

Open `notebooks/attention_images.ipynb`. Every table and figure is rendered from the committed JSON
files in `results/`, so no GPU and no retraining is needed to read it.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest              # unit tests
python -m src.train           # full grid, 13 configurations x 3 seeds
python -m src.clip_ref        # CLIP zero-shot reference
```

The grid takes roughly 11 hours on an M3 Max. Each run writes its own JSON and completed runs are
skipped, so it can be interrupted and resumed.

## Layout

| path | contents |
|---|---|
| `src/lab05.py` | Transformer encoder blocks from Lab 05 |
| `src/vit.py` | Patch embedding, pooling, classifier head |
| `src/baselines.py` | MLP and CNN from Lab 02 |
| `src/clip_ref.py` | CLIP zero-shot from Lab 10 |
| `src/data.py` | CIFAR-10 loaders and transforms from Lab 04 |
| `src/variants.py` | The experiment grid |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Metrics and figures |
| `src/attention.py` | Attention maps and head ablation |
| `results/` | One JSON per run |

Every module names the lab or slide deck each component comes from.

## License

MIT
