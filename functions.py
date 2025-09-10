import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit


def add_lagged_target(df):
    """ 
    Add a Lagged Target variable in the third position
    """
    data = df.copy()
    
    # Add a Lagged Target variable 
    data["Lagged_target"] = data.Target.shift(4)

    # Place Lagged_target in 3rd position
    cols = data.columns.tolist()
    cols.insert(2, cols.pop(cols.index('Lagged_target')))
    data = data[cols]
    data.loc[3,"Lagged_target"] = 0

    return data

# helper function to convert dataframe columns to float64 data type
def convert_float(rawdata):
    """
    Convert a panda dataset into a float64 
    """
    from pandas.api.types import is_numeric_dtype
    
    # copy the input dataframe to preserve original data
    result = rawdata.copy()
    
    # convert numeric columns to float64
    for col in result.columns:
        if is_numeric_dtype(result[col]):
            result[col] = result[col].astype("float64")
    
    # keep only columns with float64 datatype
    result = result.loc[:, [x == "float64" for x in result.dtypes]].copy() 

    return result

# prepare data for LSTM. ensure the target variable is in the first column or reorder the dataframe
def data_formatter(data, n_timesteps):

    # Convert to array
    data_f = data.select_dtypes(include=[np.number]).astype("float64")
    data_f_np = data_f.to_numpy()

    date_series = data.loc[:, 'Date'].reset_index(drop=True)

    X, y, dates = [], [], []

    for i in range(len(data_f_np) - n_timesteps + 1):
        # past 4 weeks of features
        seq_x = data_f_np[i:(i + n_timesteps), 1:]     
        # target at the END of the 4-week window
        seq_y = data_f_np[i + n_timesteps - 1, 0] 
        # Corresponding date
        date_y = date_series[i + n_timesteps - 1]
        
        # Only include if target not NaN
        if not np.isnan(seq_y):
            X.append(seq_x)
            y.append(seq_y)
            dates.append(date_y)

    X = np.array(X)
    y = np.array(y)
    dates = pd.Series(dates)

    return X, y, dates.reset_index(drop=True)




# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)  # input-hidden weights
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)  # hidden-hidden weights
            elif "bias" in name:
                # zero all biases
                param.data.fill_(0)
                # set forget gate bias to 1
                param.data[param.size(0) // 4 : param.size(0) // 2].fill_(
                    1
                )  # forget gate

        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        batch_size = x.size(0)  # x shape: (batch_size, seq_len, input_size)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(x, hidden)  # LSTM forward pass
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        output = self.fc(
            lstm_out[:, -1, :]
        )  # For sequence-to-one: use lstm_out[:, -1, :]

        return output, hidden


# Data object
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(
            [self.targets[idx]]
        )


# Training
# Training
def train_lstm(
    model,
    train_loader,
    val_loader,
    max_epochs=5000,
    learning_rate=0.001,
    patience=100,
    lr_patience=50,
    lr_decay=0.9,
    verbose=False,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=lr_decay
    )

    train_losses = []
    val_losses = []
    lr_history = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output, _ = model(data)
                val_loss += criterion(output, target).item()

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lr_history.append(scheduler.optimizer.param_groups[0]["lr"])

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save best model
        else:
            patience_counter += 1

        # Verbose
        if verbose:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1}/{max_epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.4f}"
                )

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Check if we should stop early
        if patience_counter >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch+1}! No improvement for {patience} epochs."
                )
                print(f"We take model from epoch {epoch - patience + 1}.")
                print(f"Restoring best model (val_loss: {best_val_loss:.6f})")
            model.load_state_dict(best_model_state)  # Restore best weights
            break

    return train_losses, val_losses, lr_history, best_val_loss



# Plot learning
def plot_losses_and_lr(train_losses, val_losses, lr_history):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis for losses
    ax1.plot(train_losses, label="Train Loss", linestyle="-")  # , marker="o")
    ax1.plot(val_losses, label="Val Loss", linestyle="-")  # , marker="x")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Right y-axis for learning rate
    ax2 = ax1.twinx()
    ax2.plot(lr_history, color="green", label="Learning Rate", linestyle="--")
    ax2.set_ylabel("Learning Rate")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center")

    plt.title("Training/Validation Loss and Learning Rate Schedule")
    plt.tight_layout()
    plt.show()


# Load trained model and predict
def load_and_predict(model_path, test_data):
    """Load trained model and make predictions"""

    # Load checkpoint
    checkpoint = torch.load(model_path)
    hyperparams = checkpoint["hyperparameters"]

    # Recreate model
    model = LSTM(
        hyperparams["input_size"],
        hyperparams["hidden_size"],
        hyperparams["num_layers"],
        hyperparams["output_size"],
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = []
        for data in test_data:
            data = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
            output, _ = model(data)
            predictions.append(output.cpu().numpy())

    return np.array(predictions).squeeze()


def cross_validate(X_cv, y_cv, dates_cv, lr, hidden_size_grid, num_layers_grid, dropout_grid,
                   batch_size_grid, split = 4):
    """ 
    Perform a k-fold cross-validation 
    """
    # Fix parameters 
    input_size = X_cv.shape[2]
    output_size = y_cv.shape[1] if len(y_cv.shape) > 1 else 1

    results = []
    plot_info = []
    # Grid search
    for hidden_size in hidden_size_grid:
        for num_layers in num_layers_grid:
            for d_out in dropout_grid:
                for batch_size in batch_size_grid:

                    # Array to store the loss for each fold 
                    valid_loss = []

                    # Set the cross-validation
                    tscv = TimeSeriesSplit(n_splits= split, test_size=12)
                    for i, (train_index, test_index) in enumerate(tscv.split(X_cv)):

                        # Define the model
                        model = LSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            output_size=output_size,
                            dropout=d_out
                        )

                        # Define the arrays
                        X_train, y_train, dates_train = X_cv[train_index], y_cv[train_index], dates_cv[train_index]
                        X_valid, y_valid, dates_valid = X_cv[test_index], y_cv[test_index], dates_cv[test_index]

                        # Pytorch dataset
                        train_dataset = SequenceDataset(X_train, y_train)
                        val_dataset = SequenceDataset(X_valid, y_valid)

                        # DataLoader 
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        
                        # Train model
                        train_losses, val_losses, lr_history, best_val_loss = train_lstm(
                            model,
                            train_loader,
                            val_loader,
                            learning_rate=lr,
                            patience=500,
                            lr_patience=30,
                            lr_decay=0.75,
                            max_epochs= 500
                        )

                        valid_loss.append(best_val_loss)

                        # Save model at the last 
                        if i == split-1:
                            torch.save(
                                {
                                    "model_state_dict": model.state_dict(),
                                    "hyperparameters": {
                                        "input_size": input_size,
                                        "hidden_size": hidden_size,
                                        "num_layers": num_layers,
                                        "output_size": output_size,
                                        "dropout" : d_out
                                    },
                                },
                                f"model_h{hidden_size}_l{num_layers}_d{d_out}_batch{batch_size}.pth",
                            )
                    
                    # Store average mse wrt the parameters
                    results.append({
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'dropout': d_out,
                            'batch_size' : batch_size,
                            'average_valid_loss': np.mean(valid_loss)})
                    
                    plot_info.append({
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'dropout': d_out,
                            'batch_size' : batch_size,
                            'train_loss': train_losses,
                            'valid_loss' : val_losses, 
                            'lr_history' : lr_history})
                    

    return results, plot_info  


def get_results(results, plot_info):
    """ 
    Returns the dataframe sorted and the path for the selected model. It also plot the loss of the last fold. 
    """
    # Show results
    results = pd.DataFrame(results).sort_values(by = "average_valid_loss").reset_index(drop=True)
    print(results.head(5))

    # Get best parameters
    hidden_size = results.loc[0,"hidden_size"]
    num_layers = results.loc[0,"num_layers"]
    d_out = 0
    batch_size = results.loc[0,"batch_size"]

    # Define model path
    model_pth = f"model_h{hidden_size}_l{num_layers}_d{d_out}_batch{batch_size}.pth"

    # Search in the dictionary
    best_plot_entry = next(
        entry for entry in plot_info
        if entry['hidden_size'] == hidden_size
        and entry['num_layers'] == num_layers
        and entry['dropout'] == d_out
        and entry['batch_size'] == batch_size
    )

    # Define history of the best parametrization
    train_losses = best_plot_entry['train_loss']
    val_losses = best_plot_entry['valid_loss']
    lr_history   = best_plot_entry['lr_history']

    # Plot loss <----- do not look at this 
    plot_losses_and_lr(train_losses, val_losses, lr_history)

    return results, model_pth



def predict(X_test, y_test, dates_test, model_pth):
    # Prediction
    test_predictions = load_and_predict(model_pth, X_test)

    # MSE
    mse = np.mean((y_test - test_predictions) ** 2)
    print(f"\nTest MSE: {mse}")

    # Plot test predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="True")
    plt.plot(dates_test, test_predictions, label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("LSTM Test Predictions")
    plt.show()

    return mse 

