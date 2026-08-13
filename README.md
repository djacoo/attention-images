# Which Parts of Attention Does an Image Actually Need?

An ablation study of the Vision Transformer on CIFAR-10.

Deep Learning project — Machine Learning & Deep Learning, University of Verona, A.Y. 2025-26.

## What this is

The Transformer encoder was designed for sequences of words. The Vision Transformer keeps that
encoder and swaps the tokens for image patches. This project asks which parts of the encoder still
earn their place after that swap.

A single ViT is assembled from the course lab's own encoder components, then taken apart one
component at a time — positional encoding, number of heads, depth, patch size, pooling — and each
removal is measured against the reference configuration. An MLP, a CNN and zero-shot CLIP provide
context for the numbers.

## Status

In development.

## License

MIT
