# Convolutional Neural Networks

**Convolutional Neural Networks (CNNs)** are a type of deep neural network that features specialized layers for tasks such as image classification and other computer vision tasks. They are particularly effective for images because they exploit spatial locality (nearby pixels are related) and translation invariance (a cat is a cat, regardless of its position in the image).

There are two key layers in CNNs

## Convolution Layers

Learn to identify important local patterns and relationships within the image. Here are the general steps for processing images:

1. Apply a filter (or kernel), a small grid containing learnable weights, that slides over the input image.
2. A convolution is performed by computing a weighted sum of the pixel values and filter weights at every location. This outputs a feature map capturing the filter’s activations across different regions of the image.
3. The filter slides to the next location based on the stride (step size) — typically one pixel over — performs a convolution, and the result is added to the feature map.
4. The filter continually shifts until the feature map contains activations for the entire image.
5. Multiple filters can be applied to create multiple feature maps, with each filter learning to detect different patterns (edges, textures, shapes, etc.).

## **Pooling Layers**

Reduce the spatial dimensions of feature maps while retaining the most important information through a process called *downsampling*:

1. A small window (e.g., 2x2 grid) slides across the feature map with a specific stride.
2. We pool all the values into a single, representative value (like the maximum, minimum, or average value) for each region.

Each image contains the following:

- **Pixel values**: Each pixel holds intensity values where the number of values equals the number of color channels (e.g., grayscale images have one channel and RGB images have three channels).
- **Color channels**: Most color images use three channels—Red, Green, and Blue (RGB)—where the final color of a pixel is determined by combining the values across each channel.
- **Resolution and aspect ratio**: Resolution determines the level of detail, while the aspect ratio (width vs height) affects how the image is displayed or processed. Oftentimes, we’ll need to resize or standardize image dimensions.

Using these properties of images, we’ll preprocess them using an open-sourced library called `torchvision`. This library is built on top of PyTorch, which provides us with datasets, processing tools, and pre-trained models. Preprocessing is necessary to standardize images into a consistent format that aligns with the input shape expected by pre-trained models.

We’ll preprocess the images with `torchvision` using the following sequential pipeline:

```python
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])
```

- `.Resize()`: resizes images to a fixed shape for model input
- `.ToTensor()`: converts each image into PyTorch tensors and scales values to the range `[0.0, 1.0]`
- `.Normalize()`: normalizes the pixel values within each channel based on the specified mean and standard deviation

> The normalization values shown here come from the ImageNet dataset, which was used to pre-train many popular vision models. If you’re training from scratch or using a very different dataset, calculate the mean and standard deviation from your training data.
> 

## CIFAR-10 Dataset

For our image exercises, we’ll fine-tune models to classify images in a multiclass image classification task using a popular benchmark dataset called CIFAR-10. This dataset contains 60,000 color 32x32 images, each belonging to one of 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

We’ll use `torchvision` to load the dataset and apply our preprocessing pipeline to the images:

```python
import torchvision.datasets as datasets
train = datasets.CIFAR10(root="folder", train=True, download=False, transform=transform)
test  = datasets.CIFAR10(root="folder", train=False, download=False, transform=transform)
```

- `root`: specifies the directory to store the dataset
- `train=True`: indicates the training set (`False` indicates the testing set)
- `download`: downloads the dataset if it is not already in the directory

Then, we’ll use PyTorch’s `DataLoader` to load the dataset in mini-batches:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader  = DataLoader(test,  batch_size=128, shuffle=False)
```

Batch loading helps fit the images into memory, speeds up training with GPU parallelism, and improves generalization with the random shuffling of training images.

# LightWeight Models

Here are a few pre-trained models for image tasks:

- **ResNet50** (25M Parameters): deep residual network with many convolutional, pooling, activation, and normalization layers, pre-trained on ImageNet
- **ResNet18** (11M Parameters): smaller version of ResNet50 that maintains performance, pre-trained on ImageNet
- **MobileNet_V3** (1.5M Parameters): lightweight model designed for efficiency on mobile devices, pre-trained on ImageNet

We’ll build the prediction loop similar to the MLP:

1. Pass the input batch through the network’s forward pass
2. Obtain the raw outputs (logits)
3. Convert logits to probabilities using softmax
4. Select the predicted class using argmax
5. Save the predictions for evaluation

```python
all_predictions = []
with torch.no_grad():
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        logits = model(batch_X)

        # Convert logits to predicted labels
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        
        # Save predictions
        all_predictions.append(preds.cpu().numpy())

# Join predictions
y_pred = np.concatenate(all_predictions, axis=0)
```