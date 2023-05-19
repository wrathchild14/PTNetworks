# De-noising networks

Deep learning de-noising networks using different architectures implemented in pytorch.

Intended to be used with a custom-made path tracer to make it converge faster to the rendered image of a scene.

## Encoder decoder architecture

### Results

- 30 epochs, lr=0.0001. Problems: low res
  ![](results/encdec_arch/result1.png)

## Diffusion model architecture

Whats different? Diffusion steps with ResNet blocks

In comparison
![](results/diffusion/comparison.png)

- 15 epochs, lr=0.0001
  ![](results/diffusion/result1.png)
