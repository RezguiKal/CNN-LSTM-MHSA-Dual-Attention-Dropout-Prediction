import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score
from torch.utils.data import Dataset, DataLoader
import os

# Positional Encoding for temporal sequences
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Define the hybrid CNN-LSTM-MultiHeadAttention model
class HybridModel(nn.Module):
    def __init__(self, seq_len=30, feature_dim=7, h_size=256, num_heads=16):
        super(HybridModel, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.h_size = h_size

        # Ensure h_size is divisible by num_heads
        assert h_size % num_heads == 0, "h_size must be divisible by num_heads"

        # CNN layers (treat the sequence as a 1D signal)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # LSTM layers
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=h_size, batch_first=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(h_size, max_len=seq_len)

        # Multi-Head Attention layer with increased heads and dropout
        self.multihead_attn = nn.MultiheadAttention(embed_dim=h_size, num_heads=num_heads, batch_first=True, dropout=0.2)

        # Calculate CNN output size dynamically
        cnn_out_size = self._get_cnn_output_size(seq_len * feature_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(cnn_out_size + h_size, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def _get_cnn_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.flatten().shape[0]

    def forward(self, x_cnn, x_lstm, mask=None):
        # CNN forward pass
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = self.pool1(x_cnn)
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.pool2(x_cnn)
        x_cnn = self.flatten(x_cnn)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x_lstm)  # Shape: [batch_size, seq_len, h_size]

        # Add positional encoding
        lstm_out = self.pos_encoder(lstm_out)

        # Apply padding mask to Multi-Head Attention
        if mask is not None:
            key_padding_mask = (mask == 0)  # [batch_size, seq_len], True means ignore
        else:
            key_padding_mask = None

        # Multi-Head Attention
        attn_output, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        lstm_out = torch.mean(attn_output, dim=1)  # Shape: [batch_size, h_size]

        # Concatenate CNN and LSTM outputs
        x = torch.cat((x_cnn, lstm_out), dim=1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No sigmoid, as BCEWithLogitsLoss handles it
        return x, attn_weights

# Load the dataframe
df = pd.read_csv('/content/timeseries.csv')

# Create sequences of length 30 from each enrollment_id
def create_sequences(df, seq_len=30):
    sequences = []
    labels = []
    lengths = []  # Track original sequence lengths for masking
    for enrollment_id, group in df.groupby('enrollment_id'):
        features = group.drop(['enrollment_id', 'truth'], axis=1).values
        label = group['truth'].iloc[0]
        if len(features) >= seq_len:
            for i in range(len(features) - seq_len + 1):
                sequences.append(features[i:i+seq_len])
                labels.append(label)
                lengths.append(seq_len)  # Full sequence, no padding needed
        else:
            # Pad shorter sequences with zeros
            padded = np.zeros((seq_len, features.shape[1]))
            padded[:len(features)] = features
            sequences.append(padded)
            labels.append(label)
            lengths.append(len(features))  # Record actual length
    return np.array(sequences), np.array(labels), np.array(lengths)

seq_len = 30
sequences, labels, lengths = create_sequences(df, seq_len=seq_len)
print(f"Total sequences created: {len(sequences)}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    sequences, labels, lengths, test_size=0.2, random_state=42
)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, seq_len * 7)).reshape(-1, seq_len, 7)
X_test = scaler.transform(X_test.reshape(-1, seq_len * 7)).reshape(-1, seq_len, 7)

# Verify labels are binary
if not set(y_train).issubset({0, 1}) or not set(y_test).issubset({0, 1}):
    raise ValueError("Labels must be binary (0 or 1)")

# Define a custom dataset class with padding mask
class CustomDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X_cnn = torch.tensor(X, dtype=torch.float32).transpose(1, 2).reshape(-1, 1, seq_len * 7)
        self.X_lstm = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # Create a mask: 1 for real data, 0 for padded positions
        self.mask = torch.ones(len(X), seq_len, dtype=torch.float32)
        for i, length in enumerate(lengths):
            if length < seq_len:
                self.mask[i, length:] = 0  # Set padded positions to 0

    def __len__(self):
        return len(self.X_cnn)

    def __getitem__(self, idx):
        return self.X_cnn[idx], self.X_lstm[idx], self.y[idx], self.mask[idx]

# Prepare data for training hybrid model
train_dataset = CustomDataset(X_train, y_train, lengths_train)
test_dataset = CustomDataset(X_test, y_test, lengths_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create an instance of the hybrid model
h_size = 256
num_heads = 16  # Increased from 8 to 16 (256 / 16 = 16 dimensions per head)
model = HybridModel(seq_len=seq_len, feature_dim=7, h_size=h_size, num_heads=num_heads)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function with pos_weight for class imbalance
pos_weight = torch.tensor([len(y_train[y_train == 0]) / len(y_train[y_train == 1])]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 5
counter = 0
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for inputs_cnn, inputs_lstm, labels, mask in train_loader:
        inputs_cnn, inputs_lstm, labels, mask = inputs_cnn.to(device), inputs_lstm.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs_cnn, inputs_lstm, mask)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()

        # Log gradient norms for debugging
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item()
        if total_norm == 0:
            print(f"Warning: Zero gradients at Epoch {epoch+1}")

        optimizer.step()
        train_loss += loss.item() * inputs_cnn.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs_cnn, inputs_lstm, labels, mask in test_loader:
            inputs_cnn, inputs_lstm, labels, mask = inputs_cnn.to(device), inputs_lstm.to(device), labels.to(device), mask.to(device)
            outputs, _ = model(inputs_cnn, inputs_lstm, mask)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs_cnn.size(0)
    val_loss /= len(test_loader.dataset)

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), '/content/best_model.pth')
    else:
        counter += 1
    if counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    print(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load the best model for evaluation
model.load_state_dict(torch.load('/content/best_model.pth'))

# Evaluate the model on the test set
model.eval()
predictions = []
true_labels = []
attn_weights_list = []
with torch.no_grad():
    for inputs_cnn, inputs_lstm, labels, mask in test_loader:
        inputs_cnn, inputs_lstm, labels, mask = inputs_cnn.to(device), inputs_lstm.to(device), labels.to(device), mask.to(device)
        outputs, attn_weights = model(inputs_cnn, inputs_lstm, mask)
        predictions.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        attn_weights_list.extend(attn_weights.cpu().numpy())

# Calculate metrics with optimized threshold
predictions = np.array(predictions)
true_labels = np.array(true_labels)
thresholds = np.arange(0.1, 0.5, 0.05)  # Adjusted range for better balance
best_f1, best_threshold = 0, 0
for thresh in thresholds:
    f1 = f1_score(true_labels, predictions > thresh)
    if f1 > best_f1:
        best_f1, best_threshold = f1, thresh
print(f"Best F1-score: {best_f1} at threshold {best_threshold}")

# Compute final metrics with the best threshold
final_predictions = predictions > best_threshold
f1 = f1_score(true_labels, final_predictions)
aucpr = average_precision_score(true_labels, predictions)
recall = recall_score(true_labels, final_predictions)
precision = precision_score(true_labels, final_predictions)
error_rate = 1 - np.mean(final_predictions == true_labels)  # Fixed error rate calculation

# Print predictions and true labels for the first 10 samples
print("predictions[:10], true_labels[:10]")
print(predictions[:10], true_labels[:10])

# Print attention weights for the first 7 samples (to include indices 4 and 6)
attn_weights_avg = np.mean(attn_weights_list, axis=1)  # Average across heads
print("attn_weights[:7]")
print(attn_weights_avg[:7])

# Create DataFrames for predictions and metrics
results_df = pd.DataFrame({'y_true': true_labels, 'y_pred': predictions})
metrics_df = pd.DataFrame({
    'F1-score': [f1],
    'AUC-PR': [aucpr],
    'recall': [recall],
    'precision': [precision],
    'error_rate': [error_rate],
    'best_threshold': [best_threshold]
})

# Save results
output_directory = '/content/'
os.makedirs(output_directory, exist_ok=True)
results_df.to_csv(os.path.join(output_directory, 'predictions.csv'), index=False)
metrics_df.to_csv(os.path.join(output_directory, 'metrics.csv'), index=False)

# Display results
print("F1-score:", f1)
print("AUC-PR:", aucpr)
print("Recall:", recall)
print("Precision:", precision)
print("Error Rate:", error_rate)
print(f"Results saved to {output_directory}")
