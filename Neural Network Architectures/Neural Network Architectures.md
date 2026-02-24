# Neural Network Architectures

A NNA includes different types of layers, parameters and activation functions. There are specialized architectures for various tasks:

- **Feedforward networks** called **multi-layered perceptrons (MLPs)** for tabular regression and classification tasks
- **Convolutional neural networks (CNNs)** to process images for vision tasks like image classification
- **Recurrent networks (RNNs, GRUs, and LSTMs)** to handle sequential data for sequential tasks like text generation
- **Embeddings** and **token representations** to encode text data for natural language tasks like text classification

```python
import time
import torch
import torch.nn as nn
torch.manual_seed(0)

class SimpleMLP(nn.Module):
    def __init__(self, input_size=128, hidden_size=516, output_size=1):
        super().__init__()
        self.fc1   = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP()

from custom_torchinfo import custom_summary
custom_summary(model, input_size=(64, 128))
```

Neural Network Created:

```
======================================================================
        Layer (type)              Output Shape         Param #
======================================================================
            Linear-1                 [64, 516]          66,564
              ReLU-2                 [64, 516]               0
            Linear-3                 [64, 516]         266,772
              ReLU-4                 [64, 516]               0
            Linear-5                   [64, 1]             517
         SimpleMLP-6                   [64, 1]               0
======================================================================
Total params: 333,853
Trainable params: 333,853
Non-trainable params: 0
======================================================================
```

Think of a neural network as a digital brain that learns patterns from data, much like how we learn from our experiences to speak, understand language, and recognize faces.

At its core, every neural network architecture consists of three fundamental components that work together: **layers**, **parameters**, and **forward pass**. These components determine how data flows through the network to train it and generate predictions.

Let’s define each component:

- **Layers**: The fundamental building blocks of any neural network architecture. We start with an input layer that takes in the data, hidden layers that process the data, and the output layer that generates the prediction. Each layer contains a collection of artificial neurons that process information.
- **Parameters**: There are two main parameters: *weights* and *biases*. Weights control the strength of connections between neurons across different layers, while the bias term provides more flexibility in learning complex patterns. During training, the network learns to update and adjust these parameters through backpropagation.
- **Forward Pass**: The process by which the input data travels through the network layers, being transformed by the weights, biases, and activation functions to generate an output.

## Foundational Neural Network Architectures

**Multi-Layered Perceptrons (MLPs)**: consist of fully connected layers, where every neuron in one layer connects to every neuron in the next

- Use cases: regression and classification problems involving tabular data
- Limitations: Treat all input features equally, and don’t consider spatial relationships (important for images) or temporal relationships (important for sequences).

<aside>
🧠

Tabular data is **information organized into a structured format of rows and columns, similar to a spreadsheet, CSV file, or database table**.

</aside>

**Convolutional Neural Networks (CNNs)**: Specifically designed for processing images and spatial data, using convolutional layers that contain grid-like filters to scan (slide through) the image and detect patterns.

- Use cases: computer vision tasks like image classification and object segmentation
- Strengths: captures spatial hierarchies, like detecting small patterns (edges or corners) and combining them into larger, more complex patterns (faces or objects)

**Recurrent Neural Networks**: Specifically designed for processing sequential data (like words in a sentence or time series).

- Use cases: sequential tasks like text generation, text classification, and forecasting
- Strengths: captures long-term temporal relationships within sequences

## Naural Network Architecture Components

### Feed-Froward Network

![image.png](image.png)

A multi-layer feed-forward neural network has neurons organized in layers (input, hidden, and output) with connections flowing forward from one layer to the next.

<aside>
🧠

ReLU, or [Rectified Linear Unit](https://www.google.com/search?client=opera-gx&hs=HTI&sca_esv=7154391049cab9c6&sxsrf=ANbL-n6yPuWIcvPY4tDGRc95SmzrHREmMA%3A1771941136132&q=Rectified+Linear+Unit&sa=X&ved=2ahUKEwj78fSQo_KSAxVMTDABHTlFDFoQgK4QegQIARAD&biw=2347&bih=1211&dpr=0.8), is **a non-linear activation function in deep learning.** It introduces non-linearity into the network, allowing it to learn complex patterns.

</aside>

### Sigmoid Output

![image.png](image%201.png)

A fundamental neural network unit, the sigmoid activation function (σ), transforms inputs into outputs between 0 and 1.

<aside>
🧠

The sigmoid function is important in logistic regression because **it converts the linear combination of input features into a probability between 0 and 1**. This allows the model to predict binary outcomes (e.g., yes/no) and interpret results as probabilities, making it ideal for classification tasks.

</aside>

### Convolution Operation

![image.png](image%202.png)

A convolutional kernel slides across an input grid to produce feature maps, showing how local patterns are detected through weighted connections in convolutional neural networks.

### Long-Short Term Memory (LSTM)

![image.png](image%203.png)

A single Long Short-Term Memory (LSTM) controls information flow by maintaining a memory cell state for long-term patterns and a hidden state for short-term context.