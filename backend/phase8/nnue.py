import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NNUE(nn.Module):
    """NNUE model for position evaluation."""
    def __init__(self, input_size=772, hidden_size=512):
        super(NNUE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x * 100

def load_training_data(h5_file: str, chunk_size: int = 2_000_000) -> tuple:
    """
    Load preprocessed data from HDF5 file for NNUE training.
    
    Args:
        h5_file (str): Path to HDF5 file containing features and labels.
        chunk_size (int): Number of positions to load (default: 2,000,000).
    
    Returns:
        tuple: Training and validation features and labels (X_train, X_val, y_train, y_val).
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            total_positions = f['features'].shape[0]
            load_size = min(chunk_size, total_positions)
            features = f['features'][:load_size]
            labels = f['labels'][:load_size]
        logger.info(f"Loaded {len(features)} positions from {h5_file} (total available: {total_positions})")
        if len(features) == 0:
            raise ValueError("No data found in HDF5 file")
        if features.shape[1] != 772:
            raise ValueError(f"Expected 772 features, got {features.shape[1]}")
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except FileNotFoundError:
        logger.error(f"File {h5_file} not found")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def train_nnue(h5_file: str, epochs: int = 50, batch_size: int = 1024, 
               save_path: str = "b:\\chess_engine_4\\main\\nnue_model.pth"):
    """
    Train the NNUE model using preprocessed HDF5 data.
    
    Args:
        h5_file (str): Path to HDF5 file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        save_path (str): Path to save trained model.
    """
    try:
        X_train, X_val, y_train, y_val = load_training_data(h5_file)
        logger.info(f"Training data: {len(X_train)} samples, Validation data: {len(X_val)} samples")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NNUE().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(device))
                val_loss = criterion(val_outputs, y_val.to(device))
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader):.6f}, "
                       f"Val Loss: {val_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        return model
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise