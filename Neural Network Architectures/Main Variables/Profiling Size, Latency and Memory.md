# Profiling Size, Latency and Memory

One fundamental aspect is choosing the right model **size**, this means balancing between performance and practicality:

- A larger model might achieve higher accuracy but require more resources and computing power, leading to slower inference times.
- A smaller model may sacrifice some accuracy but offer faster predictions and be easier to deploy.

```python
model_params = list(model.parameters()) + list(model.buffers())
size = 0
for t in model_params:
    size += t.numel() * t.element_size()
```

- `model_params`: returns all trainable parameters from `model.parameters()` (like weights and biases), and non-trainable parameters from `model.buffers()` (like normalization layers)
- For each tensor `t`: Multiply the total number of elements in the tensor from `t.numel()` by the size of each element in bytes from `t.element_size()`
- `size`: accumulates and yields the model’s total size in bytes

---

**Latency** refers to the time it takes for a model to process input data through its forward pass and generate an output.

Latency is a critical factor for system design and user experience:

- Lower latency ensures fast, responsive predictions, especially in real-time applications (such as for self-driving applications).
- Reduce latency with smaller models or optimized architectures, which may reduce accuracy.

To measure latency, we’ll record the time before and after running the model on synthetic input data `x` and then average across multiple iterations (e.g., 50 iterations):

```python
model.eval()
start = time.perf_counter()
iters = 50
with torch.inference_mode():
    for _ in range(iters):
        _ = model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
elapsed = time.perf_counter() - start
latency = elapsed / iters
```

- Set the model to evaluation mode with `model.eval()`, and turn off gradient tracking with `torch.inference_mode()` (uses less memory).
- `torch.cuda.synchronize()`: ensures that all GPU operations finish before recording time
- `time.perf_counter()`: measures the elapsed time
- `latency`: returns the average latency per forward pass by dividing the total `elapsed` time by the number of iterations

---

**Memory usage** refers to the amount of system or GPU memory consumed while running the model inference and generating predictions with the forward pass

Managing memory is essential for assessing hardware limitations for efficient training and deployment:

- Reduce memory usage with smaller batch sizes, or use smaller models or optimized architectures, which may impact performance and training speed.

To measure (GPU) memory, we’ll first need to reset memory tracking, then run the model and inspect the memory allocated:

```python
torch.cuda.empty_cache()                 
torch.cuda.reset_peak_memory_stats()

model.eval()
with torch.inference_mode():
    y = model(x)
torch.cuda.synchronize()
current = torch.cuda.memory_allocated() / (1024**2)
peak    = torch.cuda.max_memory_allocated() / (1024**2)
```

- `torch.cuda.empty_cache()`: clears any unused cached memory blocks, and `torch.cuda.reset_peak_memory_stats()` resets PyTorch’s internal tracker
- `torch.cuda.memory_allocated()`: reports the memory currently in use
- `torch.cuda.max_memory_allocated()`: reports the peak memory used during the forward pass