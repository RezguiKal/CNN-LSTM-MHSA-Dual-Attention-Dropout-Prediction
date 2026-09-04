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
import matplotlib.pyplot as plt
import shap

# Positional Encoding for temporal sequences with adjusted decay
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(1000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Define the hybrid CNN-LSTM-DualAttention model with weighted combination
class HybridModel(nn.Module):
    def __init__(self, seq_len=30, feature_dim=7, h_size=256, num_heads=16):
        super(HybridModel, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.h_size = h_size
        self.split_point = seq_len // 2  # Split at time step 15

        # Ensure h_size is divisible by num_heads
        assert h_size % num_heads == 0, "h_size must be divisible by num_heads"

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # LSTM layers
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=h_size, batch_first=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(h_size, max_len=seq_len)

        # Dual Multi-Head Attention layers
        self.early_attn = nn.MultiheadAttention(embed_dim=h_size, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.late_attn = nn.MultiheadAttention(embed_dim=h_size, num_heads=num_heads, batch_first=True, dropout=0.2)

        # Learnable weights for combining early and late attention outputs
        self.combination_weights = nn.Linear(h_size * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

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
        lstm_out, _ = self.lstm(x_lstm)

        # Add positional encoding
        lstm_out = self.pos_encoder(lstm_out)

        # Split the sequence into early and late parts
        early_out = lstm_out[:, :self.split_point, :]  # Time steps 0-14
        late_out = lstm_out[:, self.split_point:, :]   # Time steps 15-29

        # Apply padding mask to Multi-Head Attention
        if mask is not None:
            early_mask = mask[:, :self.split_point]  # [batch_size, 15]
            late_mask = mask[:, self.split_point:]   # [batch_size, 15]
            early_key_padding_mask = (early_mask == 0)  # True means ignore
            late_key_padding_mask = (late_mask == 0)
        else:
            early_key_padding_mask = None
            late_key_padding_mask = None

        # Early Multi-Head Attention (time steps 0-14)
        early_attn_output, early_attn_weights = self.early_attn(
            early_out, early_out, early_out, key_padding_mask=early_key_padding_mask
        )
        early_attn_output = torch.mean(early_attn_output, dim=1)  # [batch_size, h_size]

        # Late Multi-Head Attention (time steps 15-29)
        late_attn_output, late_attn_weights = self.late_attn(
            late_out, late_out, late_out, key_padding_mask=late_key_padding_mask
        )
        late_attn_output = torch.mean(late_attn_output, dim=1)  # [batch_size, h_size]

        # Learnable weighted combination of early and late attention outputs
        combined = torch.cat((early_attn_output, late_attn_output), dim=-1)  # [batch_size, h_size * 2]
        weights = self.combination_weights(combined)  # [batch_size, 2]
        weights = self.softmax(weights)  # Normalize weights
        w_early, w_late = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)  # [batch_size, 1]
        combined_attn_output = w_early * early_attn_output + w_late * late_attn_output  # [batch_size, h_size]

        # Concatenate CNN and combined attention outputs
        x = torch.cat((x_cnn, combined_attn_output), dim=1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No sigmoid, as BCEWithLogitsLoss handles it

        # Return outputs and both sets of attention weights
        return x, (early_attn_weights, late_attn_weights)

# Load the dataframe
df = pd.read_csv('/content/timeseries.csv')

# Extract original feature names (excluding 'enrollment_id' and 'truth')
feature_names = df.drop(['enrollment_id', 'truth'], axis=1).columns.tolist()
print("Original feature names:", feature_names)

# Create sequences of length 30 from each enrollment_id
def create_sequences(df, seq_len=30):
    sequences = []
    labels = []
    lengths = []
    for enrollment_id, group in df.groupby('enrollment_id'):
        features = group.drop(['enrollment_id', 'truth'], axis=1).values
        label = group['truth'].iloc[0]
        if len(features) >= seq_len:
            for i in range(len(features) - seq_len + 1):
                sequences.append(features[i:i+seq_len])
                labels.append(label)
                lengths.append(seq_len)
        else:
            padded = np.zeros((seq_len, features.shape[1]))
            padded[:len(features)] = features
            sequences.append(padded)
            labels.append(label)
            lengths.append(len(features))
    return np.array(sequences), np.array(labels), np.array(lengths)

seq_len = 30
sequences, labels, lengths = create_sequences(df, seq_len=seq_len)
print(f"Total sequences created: {len(sequences)}")

# Create feature names for SHAP (time_i_feature_name)
shap_feature_names = [f'time_{i}_{feat}' for i in range(seq_len) for feat in feature_names]

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
        self.mask = torch.ones(len(X), seq_len, dtype=torch.float32)
        for i, length in enumerate(lengths):
            if length < seq_len:
                self.mask[i, length:] = 0

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
num_heads = 16
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

# Evaluate the model on the test set with advanced XAI
model.eval()
predictions = []
true_labels = []
early_attn_weights_list = []
late_attn_weights_list = []
raw_features_list = []

# First pass: Collect predictions and attention weights without gradients
with torch.no_grad():
    for inputs_cnn, inputs_lstm, labels, mask in test_loader:
        inputs_cnn, inputs_lstm, labels, mask = inputs_cnn.to(device), inputs_lstm.to(device), labels.to(device), mask.to(device)
        outputs, (early_attn_weights, late_attn_weights) = model(inputs_cnn, inputs_lstm, mask)
        predictions.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        early_attn_weights_list.extend(early_attn_weights.cpu().numpy())
        late_attn_weights_list.extend(late_attn_weights.cpu().numpy())
        raw_features_list.extend(inputs_lstm.cpu().numpy())

predictions = np.array(predictions)
true_labels = np.array(true_labels)
early_attn_weights_avg = np.array(early_attn_weights_list)
late_attn_weights_avg = np.array(late_attn_weights_list)

# Use a fixed threshold of 0.35 for classification
threshold = 0.35
final_predictions = predictions > threshold
f1 = f1_score(true_labels, final_predictions)
aucpr = average_precision_score(true_labels, predictions)
recall = recall_score(true_labels, final_predictions)
precision = precision_score(true_labels, final_predictions)
error_rate = 1 - np.mean(final_predictions == true_labels)

# Print predictions and true labels for the first 10 samples
print("predictions[:10], true_labels[:10]")
print(predictions[:10], true_labels[:10])

# Print attention weights and raw features for the first 7 samples
print("early_attn_weights[:7]")
print(early_attn_weights_avg[:7])
print("late_attn_weights[:7]")
print(late_attn_weights_avg[:7])
print("raw_features[:7, :, 0]  # First feature for first 7 samples")
print(np.array(raw_features_list)[:7, :, 0])

# XAI: Attention-based visualization for both early and late attention
sample_idx = 0
early_attn_weights_sample = early_attn_weights_avg[sample_idx]
early_attn_weights_sample_avg = np.mean(early_attn_weights_sample, axis=0)
late_attn_weights_sample = late_attn_weights_avg[sample_idx]
late_attn_weights_sample_avg = np.mean(late_attn_weights_sample, axis=0)

# Plot early and late attention weights separately
plt.figure(figsize=(10, 6))
plt.plot(range(15), early_attn_weights_sample_avg, label='Early Attention Weights (Time Steps 0-14)')
plt.plot(range(15, 30), late_attn_weights_sample_avg, label='Late Attention Weights (Time Steps 15-29)')
plt.title(f'Attention Weights for Sample {sample_idx}')
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.legend()
plt.grid(True)
plt.savefig(f'/content/dual_attention_sample_{sample_idx}.png')
plt.close()

# XAI: SHAP analysis
def model_predict(X):
    X = X.reshape(-1, seq_len, 7)  # Reshape flat input back to [batch_size, seq_len, feature_dim]
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    X_cnn = X_tensor.transpose(1, 2).reshape(-1, 1, seq_len * 7)
    mask = torch.ones(X_tensor.size(0), seq_len, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs, _ = model(X_cnn, X_tensor, mask)
    probs = torch.sigmoid(outputs).cpu().numpy()  # Shape: [batch_size, 1]
    return probs.flatten()  # Shape: [batch_size]

# Prepare background dataset (using training data)
background = X_train[:20].reshape(20, -1)  # Flatten to [20, 210]
test_subset = X_test[:5].reshape(5, -1)  # [5, 210]
explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(test_subset)
shap.summary_plot(shap_values, test_subset, feature_names=shap_feature_names, show=False)
plt.savefig(f'/content/shap_summary.png')
plt.close()

print(f"XAI visualizations saved to /content/")

# Create DataFrames for predictions and metrics
results_df = pd.DataFrame({'y_true': true_labels, 'y_pred': predictions})
metrics_df = pd.DataFrame({
    'F1-score': [f1],
    'AUC-PR': [aucpr],
    'recall': [recall],
    'precision': [precision],
    'error_rate': [error_rate],
    'threshold': [threshold]
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
