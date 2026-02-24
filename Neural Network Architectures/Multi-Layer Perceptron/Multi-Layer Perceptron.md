# Multi-Layer Perceptron

**Multi-Layer Perceptrons (MLPs)** are *feedforward networks* in which information flows in one direction: starting from the input layer, then through one or more hidden layers, and lastly the output layer. MLPs consist of fully connected layers where each neuron connects to every neuron in the next layer.

As the data flows through multiple hidden layers, the network learns complex patterns by applying transformations using non-linear **activation functions**.

The most widely used activation function is the **Rectified Linear Unit (ReLU)** activation function, which zeros out negative values while keeping positive values.

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1   = nn.Linear(input_size, hidden_size)
        self.fc2   = nn.Linear(hidden_size, hidden_size)
        self.fc3   = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
```

In the code block, we created:

- The custom PyTorch model class using `nn.Module` initialized with three input parameters:
    - `input_size`: number of neurons equal to the number of input features
    - `hidden_size`: number of neurons in the hidden layers
    - `output_size`: number of neurons in the output layer equal to the number of desired outputs
- Three fully-connected layers are defined using `nn.Linear()`.
    - `fc1`: the first hidden layer that transforms the input features into hidden features
    - `fc2`: the second hidden layer that continually processes the hidden features
    - `fc3`: the output layer that produces the network’s final outputs (logits)
- The `forward` [method](https://www.codecademy.com/resources/docs/general/method) defines the forward pass where the ReLU activation is applied to the outputs for the `fc1` and `fc2` layers.

## **Universal Approximation**

Now that you know the main components of neural networks, let’s tie everything together and examine an interesting property of MLPs called the **Universal Approximation Theorem**. The theorem states that an MLP with at least one hidden layer and a non-linear activation function can approximate any continuous function mapping the inputs to the target output.

Think of it like this: The theorem tells you that somewhere in the vast space of possible neural network configurations, there’s one that can solve your problem perfectly. But finding it is like searching for a needle in a haystack the size of the universe, that’s where all the practical work of architecture design, training techniques, and optimization comes in.

In practice, training an MLP to learn the relationship between the inputs and outputs involves three components:

### Loss Function

Measures how far the network’s predictions are from the actual targets

```python
import torch.nn as nn

# Regression — Mean Squared Error 
loss_fn = nn.MSELoss()

# Classification — Binary Cross-Entropy
loss_fn = nn.BCEWithLogitsLoss()
```

### **Optimizer**

A method for adjusting the model’s [parameter](https://www.codecademy.com/resources/docs/general/parameter) weights during training to reduce the loss function.

```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training Loop

Iterations (epochs) over batches of the training dataset, where in each iteration we:

1. Pass the input batch through the network’s forward pass
2. Obtain the raw outputs (logits) and compute the loss
3. Perform a backward pass (backpropagation)
4. Use the optimizer to update the weights to reduce the loss
5. Track training metrics like the average loss

```python
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)

        # Compute loss
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        num_batches += 1

    # Average loss
    avg_loss = epoch_loss / num_batches
```

- First, we select the number of [epochs](https://www.codecademy.com/resources/docs/general/machine-learning/epochs), where a single epoch is one complete pass through the entire training dataset.
    
    Preview: Docs Loading link description
    
- The model is set to training mode using `.train()` to enable training behavior like updating weights.
- We train with smaller *batches* of the training dataset instead of the full dataset because it balances efficiency (the full dataset might not fit into memory), stability (better convergence), and generalization (may help prevent overfitting).