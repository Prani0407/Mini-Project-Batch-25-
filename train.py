import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score


# Function to train model
def train_fn(model, epochs, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        # Calculate average validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        

    print(" Training finished")

    # Plot the training loss and validation loss for each epoch
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses,label='Validation Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model

# Function to evaluate model and calculate metrics
def evaluate_fn(stock, type, model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, val_loader, scaler):
    # Set the model to evaluation mode
    model.eval()
    val_predicted_values = []
    val_true_values = []
    # Disable gradient calculation since we are in evaluation mode
    with torch.no_grad():

        # Predict on the training data
        train_predictions = model(X_train_tensor)
        train_predicted_values = train_predictions.squeeze().numpy()
        # Evaluate on the validation data
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_predicted_values.extend(outputs.squeeze().numpy())
            val_true_values.extend(targets.numpy())
        # Predict on the test data
        predictions = model(X_test_tensor)
        predicted_values = predictions.squeeze().numpy()
    
    # Convert true train target values tensor to numpy array
    train_true_values = y_train_tensor.numpy()
    # Convert true target values tensor to numpy array
    true_values = y_test_tensor.numpy()

    # Inverse transform the predicted values
    train_predicted_values_original = scaler.inverse_transform(train_predicted_values.reshape(-1, 1)).flatten()
    # Inverse transform the true target values
    train_true_values_original = scaler.inverse_transform(train_true_values.reshape(-1, 1)).flatten()
    val_predicted_values_original = scaler.inverse_transform(np.array(val_predicted_values).reshape(-1, 1)).flatten()
    val_true_values_original = scaler.inverse_transform(np.array(val_true_values).reshape(-1, 1)).flatten()
    predicted_values_original = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    true_values_original = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()

    # Metrics
    rmse = (np.mean((predicted_values - true_values) ** 2))**0.5
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    mse = np.mean((predicted_values - true_values) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    mae = np.mean(np.abs(predicted_values - true_values))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    r_squared = r2_score(true_values, predicted_values)
    print(f"R-squared: {r_squared:.4f}")

    # Plot the actual vs. predicted train values
    plt.figure(figsize=(12, 6))
    plt.plot(train_true_values_original, label="Actual Training Values", color='b')
    plt.plot(train_predicted_values_original, label="Predicted Training Values", color='r')
    plt.title('Actual vs. Predicted Training data Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Plot the actual vs. predicted validation values
    plt.figure(figsize=(12, 6))
    plt.plot(val_true_values_original, label="Actual Validation Values", color='b')
    plt.plot(val_predicted_values_original, label="Predicted Validation Values", color='r')
    plt.title('Actual vs. Predicted Validation data Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Plot the actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(true_values_original, label="Actual Values", color='b')
    plt.plot(predicted_values_original, label="Predicted Values", color='r')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    print("Actual test data price values",true_values_original)
    print("Predicted test data price values",predicted_values_original)

    # Create a DataFrame with two columns: 'Actual' and 'Predicted'
    df = pd.DataFrame({
        'Actual': true_values_original,
        'Predicted': predicted_values_original
    })

    filepath=f"{stock} {type} model pred_vs_actual.csv"

    # Optionally, save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)
