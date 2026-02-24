# Generating Inferences

After training is complete, the MLP can be applied to new data using inference. Inference is the process by which a trained network uses its learned knowledge to make predictions on new data.

Inference involves two key steps:

- Create a prediction loop to input the data through the forward pass.
- Generate and interpret the predictions.

But first, let’s cover how to save and load your trained model.

### **Saving and Loading Model Parameters**

Saving and loading a model’s learned parameters is advantageous because it allows us to use trained models without retraining from scratch or using pre-trained models trained elsewhere.

PyTorch allows us to save and load trained model parameters (learnable weights and biases) using `state_dict`:

```python
# Save parameters
torch.save(model.state_dict(), "mlp_weights.pt")

# Load parameters
state_dict = torch.load(
    "mlp_weights.pt",
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
```

> `*state_dict` doesn’t store the model class, so we have to ensure that the model architecture we’re loading the weights into matches.*
> 

### **Prediction Loop**

The prediction loop is similar to the training loop in that we pass the new data through the trained network to obtain raw outputs (logits). The difference is that we disable gradient updates to prevent updating the parameters.

Iterations (epochs) over batches of the new dataset, where in each iteration we:

1. Pass the input batch through the network’s forward pass
2. Obtain the raw outputs (logits) and optionally compute the loss for evaluation
3. Convert logits to predictions and track metrics like accuracy

```python
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import numpy as np

def predict(model, dataloader, device="cpu"):
    model.eval() 
    loss_fn = nn.BCEWithLogitsLoss() 
    
    test_loss = 0.0
    num_batches = 0
    all_predictions, all_targets = [], []

    with torch.no_grad(): 
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)

            test_loss += loss.item()
            num_batches += 1

            # Convert logits to predicted labels
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_predictions.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    avg_loss = test_loss / max(num_batches, 1)
    y_true = np.concatenate(all_targets).ravel()
    y_pred = np.concatenate(all_predictions).ravel()
    acc = accuracy_score(y_true, y_pred)
    
    return avg_loss, acc, y_true, y_pred

# Run prediction loop
avg_loss, acc, y_true, y_pred = predict(mlp_relu, test_loader, device=device)
```

- We set the model to evaluation mode using `.eval()` and use `torch.no_grad()` to disable gradient computation.
- The logits (raw outputs) are transformed into predicted probabilities and labels.
- Since we’re solving a classification task, we print out the accuracy using the `accuracy_score()` function from Scikit-learn.
- The function returns the average loss, accuracy, true labels, and predicted labels.

### **Evaluate the Outputs**

Once we’ve generated predictions, the next step is to evaluate the outputs.

For classification tasks, we want to know how many observations the model predicted correctly and how well it performed across classes in the dataset. We’ll use the `classification_report` from Scikit-learn:

```python
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=['Not Canceled', 'Canceled'])
```

Key points when interpreting the results:

- **Average Loss**: measures how far the model’s predictions are from the true label on average
- **Accuracy**: the proportion of correctly classified samples over all predictions
- **Classification Report**: breaks down the precision, recall, and F1-score performance per class
    - **Precision**: Measures the false positive performance — of all observations the model predicted as positive, how many were actually positive?
    - **Recall**: Measures the false negative performance — of all actual positives, how many did the model correctly identify?
    - **F1-Score**: harmonic mean of precision and recall (balanced measure)