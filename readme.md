# De-noising networks

Deep learning de-noising networks using different architectures implemented in pytorch.

Intended to be used with a custom-made path tracer to make it converge faster to the rendered image of a scene.

## U-net

### Results

Trains really quickly. **Note**: will re-train on a more noisy data. 

|               Before                |               After                |
|:-----------------------------------:|:----------------------------------:|
| ![](results/pt_results/before1.jpg) | ![](results/pt_results/after1.jpg) |
| ![](results/pt_results/noisy.jpg)  | ![](results/pt_results/denoised.png)|

## Encoder decoder architecture

Failed architecture because of how un-precise the images get predicted.

### Results

In comparison
![](results/encdec/comparison.png)
