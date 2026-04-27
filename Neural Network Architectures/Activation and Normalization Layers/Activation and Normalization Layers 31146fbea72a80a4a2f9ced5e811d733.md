# Activation and Normalization Layers

## Activation Functions

The key component that enables neural networks to learn complex patterns beyond linear relationships is the use of *non-linear* activation functions between layers.

While ReLU is considered the most widely used activation function, recent papers have introduced alternatives like GELU and Swish that may improve performance depending on your task and dataset:

**ReLU (Rectified Linear Unit `f(x)=max(0,x)`)**

Zeros out negative values while keeping positive values.

- Simple, computationally efficient, and can help reduce vanishing gradients
- Can suffer from the dying ReLU problem when neurons only output zero

**GELU (Gaussian Error Linear Unit)**

Scales inputs based on how much greater they are than other inputs, using a smooth approximation.

- The exact formula involves the cumulative distribution function of the standard normal distribution, but PyTorch uses an efficient approximation.
- Smooth, differentiable, and especially strong in transformer architectures like BERT and GPT
- Slightly more computationally expensive than ReLU

> PyTorch calls this SiLU (Sigmoid Linear Unit).
> 

## Normalization Layers

Neural networks can also benefit from **normalization layers** that rescale and shift the outputs from activations. Normalizing intermediate values stabilizes training, which leads to faster training and sometimes improved generalization performance.

There are multiple types of normalization, but here are the two most common types:

### **BatchNorm**

Each feature is normalized using its mean and variance computed over the entire batch.

- Commonly used in feedforward MLPs and CNNs
- Works best with reasonably large batch sizes
- Small batches may result in noisy estimates

### **LayerNorm**

Each row is normalized by computing the mean and variance of all features in that row.

- Commonly used in RNNs and transformers
- Works consistently regardless of batch size since it doesn’t depend on batch statistics
- Slightly more expensive computationally due to per-sample calculations

## TL;DR

- **Activation functions** introduce non-linearity, allowing the network to learn complex patterns in the data.
- **Normalization layers** stabilize these activations, helping gradients flolw smoothly and improve training.

In practice, we often combine these components in the following order: 

<aside>
🧠

Linear transformation → Normalization → Activation

</aside>

```python
nn.Sequential(
    nn.Linear(128, 64, bias=False),
    nn.BatchNorm1d(64),
    nn.ReLU())
```