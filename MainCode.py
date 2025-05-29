import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from bs4 import BeautifulSoup
import requests
import json
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import statsmodels.api as sm
import optuna
from optuna.samplers import TPESampler


# === FILE SAMPLING + DATA EXTRACTION ===

def load_sp500_data(filepath):
    """
    Load S&P 500 stock data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing SP500 data
        
    Returns:
        dict: Dictionary with (symbol, date) tuples as keys and stock data as values
    """
    
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)

        df = df.rename(columns={
            'date': 'Date',
            'ticker': 'Ticker',
            'comnam': 'comnam',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',  
            'market_cap': 'MarketCap'})

        # Convert date column to datetime.date objects
        # Assuming date is in 'YYYY-MM-DD' format - adjust as needed
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Create a dictionary with (symbol, date) as key and a dictionary of stock data as value
        sp500_dict = {}
        
        for _, row in df.iterrows():
            symbol = row['Ticker']
            date = row['Date']
            
            # Create a key from symbol and date
            key = (symbol, date)
            
            # Create a dictionary of the row data
            row_dict = dict(row)
            
            # Use MarketCap for MarketValue if it exists
            if 'MarketCap' in row_dict:
                row_dict['MarketValue'] = row_dict['MarketCap']
            
            # Store all available data for this symbol and date
            sp500_dict[key] = row_dict
        
        print(f"Loaded {len(df)} stock data points for {df['Ticker'].nunique()} symbols")
        return sp500_dict
        
    except Exception as e:
        print(f"Error loading SP500 data: {str(e)}")
        return {}

def calculate_returns(sp500_data, symbol, date, periods=[1, 30]):
    """
    Calculate returns for a stock over specified periods. If the date isn't available,
    it finds the closest previous date.

    Args:
        sp500_data (dict): Dictionary with SP500 data
        symbol (str): Stock symbol
        date (datetime.date): Reference date
        periods (list): List of periods (in days) to calculate returns for

    Returns:
        dict: Dictionary with returns for each period
    """

    returns = {}

    # Find closest past date if exact date is not available
    ref_date = date
    for days_back in range(10):  # Look back up to 10 days
        ref_key = (symbol, ref_date)
        if ref_key in sp500_data:
            try:
                ref_price = float(sp500_data[ref_key]['Open'])
                if not (np.isnan(ref_price) or ref_price <= 0):
                    break
            except (ValueError, TypeError, KeyError):
                pass
        ref_date -= datetime.timedelta(days=1)
    else:
        # No suitable past date found
        return {f"{p}D_return": None for p in periods}

    # Calculate returns for each period
    for period in periods:
        target_date = ref_date + datetime.timedelta(days=period)
        found_future_data = False
        for offset in range(5):  # Try up to 5 days forward to find data
            check_date = target_date + datetime.timedelta(days=offset)
            check_key = (symbol, check_date)
            if check_key in sp500_data:
                try:
                    target_price = float(sp500_data[check_key]['Open'])
                    if np.isnan(target_price) or target_price <= 0:
                        continue
                    period_return = (target_price - ref_price) / ref_price
                    returns[f"{period}D_return"] = float(period_return)
                    found_future_data = True
                    break
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
        if not found_future_data:
            returns[f"{period}D_return"] = None

    return returns

def extract_target_text(filepath, sp500_data):
    """
    Extract text from an 8-K filing and enhance with SP500 data.
    
    Args:
        filepath (str): Path to the 8-K HTM file
        sp500_data (dict): Dictionary with SP500 data
        
    Returns:
        dict: Dictionary with extracted text and financial data
    """

    filename = os.path.basename(filepath)
    symbol, cik, date_str = filename.replace('.htm', '').split('_')
    
    # Convert the date string to an actual date object
    # Assuming the date format is YYYYMMDD
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()

    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    full_text = soup.get_text(separator='\n', strip=True)

    start_str = "Pre-commencement communications pursuant to Rule 13e-4(c) under the Exchange Act (17 CFR 240.13e-4(c))"
    end_str = "SIGNATURE"

    try:
        start_idx = full_text.index(start_str) + len(start_str)
        end_idx = full_text.index(end_str)
        target_text = full_text[start_idx:end_idx].strip()
    except ValueError:
        target_text = full_text
    
    # Calculate returns using SP500 data
    returns = calculate_returns(sp500_data, symbol, date)
    
    # Initialize market_value with a default
    market_value = None
    
    # Get MarketValue from sp500_data if available
    key = (symbol, date)
    if key in sp500_data and 'MarketValue' in sp500_data[key]:
        market_value = sp500_data[key]['MarketValue']
    
    # If we couldn't get the market value from sp500_data, use random value
    if market_value is None:
        market_value = None
        # print(f"Warning: MarketValue not found for {symbol} on {date}, using random value")
    
    # Create result dictionary
    result = {
        'symbol': symbol,
        'cik': cik,
        'date': date,
        'text': target_text,
        'MarketValue': market_value
    }
    
    # Add returns
    result.update(returns)
    
    return result


def sample_htm_files(directory, sp500_data, sample_size=None):
    """
    Sample HTM files and extract text with SP500 data.
    
    Args:
        directory (str): Directory containing HTM files
        sp500_data (dict): Dictionary with SP500 data
        sample_size (int): Number of files to sample (None for all)
        
    Returns:
        dict: Dictionary with extracted text and financial data
    """

    all_files = [f for f in os.listdir(directory) if f.endswith('.htm')]
    
    # If sample size is None or greater than available files, use all files
    if sample_size is None or sample_size >= len(all_files):
        sample_files = all_files
        print(f"Processing all {len(sample_files)} HTM files...")
    else:
        sample_files = random.sample(all_files, sample_size)
        print(f"Sampling {len(sample_files)} HTM files...")
    
    result_dict = {}
    missing_data_count = 0
    
    for filename in tqdm(sample_files, desc="Extracting text"):
        filepath = os.path.join(directory, filename)
        entry = extract_target_text(filepath, sp500_data)
        
        # Check if we have return data
        if entry['1D_return'] is None and entry['30D_return'] is None:
            missing_data_count += 1
            # Skip entries with no return data if too many are missing
            if missing_data_count > len(sample_files) * 0.5:  # If more than 50% missing
                continue
        
        result_dict[filename] = entry
    
    print(f"Processed {len(result_dict)} files ({missing_data_count} had missing return data)")
    return result_dict

def filter_missing_returns(data_dict):
    """
    Remove entries from the data dictionary with missing or non-numeric 1D_return, 30D_return, or MarketValue.
    
    Args:
        data_dict (dict): Dictionary containing data with return values and market values
        
    Returns:
        dict: Filtered dictionary with complete numeric return data and market values
    """
    filtered_dict = {}
    missing_returns_count = 0
    missing_market_value_count = 0
    total_removed = 0

    for key, entry in data_dict.items():
        # Check for valid market value
        try:
            market_value = float(entry.get('MarketValue'))
            if np.isnan(market_value) or market_value <= 0:
                missing_market_value_count += 1
                total_removed += 1
                continue
        except (TypeError, ValueError):
            missing_market_value_count += 1
            total_removed += 1
            continue
            
        # Check for valid return values
        try:
            ret_1d = float(entry['1D_return'])
            ret_30d = float(entry['30D_return'])
            if np.isnan(ret_1d) or np.isnan(ret_30d):
                missing_returns_count += 1
                total_removed += 1
                continue
        except (TypeError, ValueError):
            missing_returns_count += 1
            total_removed += 1
            continue

        # Entry has valid market value and return values
        filtered_dict[key] = {
            **entry,
            'MarketValue': market_value,
            '1D_return': ret_1d,
            '30D_return': ret_30d
        }

    print(f"Filtered out {total_removed} entries:")
    print(f"- {missing_returns_count} with missing or non-numeric return values")
    print(f"- {missing_market_value_count} with missing or non-numeric market values")
    print(f"Remaining entries: {len(filtered_dict)}")

    return filtered_dict


# === TEXT VECTORIZATION ===

def get_optimal_device():
    """
    Detect the optimal device for PyTorch on macOS, Windows, or Linux systems.
    Returns the best available device for the current platform.
    """
    if torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) GPU acceleration
        return torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU
        return torch.device("cuda")
    else:
        # CPU fallback
        return torch.device("cpu")

def vectorize_text(texts, model_type="ollama", batch_size=32, ollama_host="http://localhost:11434"):
    """
    Vectorize a list of texts using Ollama, FinBERT, or BERT.

    Args:
        texts (list): List of strings to vectorize
        model_type (str): 'ollama', 'finbert', or 'bert'
        batch_size (int): Batch size for Ollama (ignored for BERT/FinBERT)
        ollama_host (str): Ollama host URL

    Returns:
        np.ndarray: Array of embeddings
    """

    if model_type == "ollama":
        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        print(f"Generating embeddings with Ollama for {len(texts)} documents...")

        for batch_texts in tqdm(batches, desc="Embedding batches"):
            batch_embeddings = []
            for text in tqdm(batch_texts, desc="Documents in batch", leave=False):
                url = f"{ollama_host}/api/embeddings"
                headers = {"Content-Type": "application/json"}
                data = {"model": "nomic-embed-text:latest", "prompt": text}

                try:
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    response.raise_for_status()
                    embedding = response.json().get("embedding", [])
                    if embedding:
                        batch_embeddings.append(embedding)
                    else:
                        tqdm.write(f"Warning: Empty embedding for: {text[:50]}")
                        if batch_embeddings:
                            dim = len(batch_embeddings[0])
                            batch_embeddings.append([0.0] * dim)
                        else:
                            batch_embeddings.append([])
                except Exception as e:
                    tqdm.write(f"Error getting embedding: {str(e)}")
                    batch_embeddings.append([])

            if batch_embeddings:
                dim = next((len(emb) for emb in batch_embeddings if emb), 0)
                batch_embeddings = [emb if emb else [0.0] * dim for emb in batch_embeddings]
                all_embeddings.append(np.array(batch_embeddings, dtype=np.float32))

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    elif model_type in {"finbert", "bert"}:

        model_name = "yiyanghkust/finbert-pretrain" if model_type == "finbert" else "bert-base-uncased"
        device = get_optimal_device()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        print(f"Generating embeddings with {model_type.upper()} on device {device} for {len(texts)} documents...")
        embeddings = []

        with torch.no_grad():
            for text in tqdm(texts, desc=f"{model_type.upper()} Encoding"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked = last_hidden_state * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                mean_pooled = summed / counts
                embeddings.append(mean_pooled.squeeze().cpu().numpy())

        return np.vstack(embeddings)

    else:
        raise ValueError("model_type must be 'ollama', 'finbert', or 'bert'")

# === PREDCTION MODELS === 

class FinancialTransformer(nn.Module):
    """
    Transformer-based model for financial return prediction from text embeddings.
    Predicts both 1-day and 30-day returns simultaneously.
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(FinancialTransformer, self).__init__()
        
        # Dimensionality reduction layer - convert embedding to standard size
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Position-wise feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output layers - one for each target
        self.output_1d = nn.Linear(hidden_dim, 1)  # 1-day return prediction
        self.output_30d = nn.Linear(hidden_dim, 1)  # 30-day return prediction

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Since we're working with embeddings and not sequences, we need to
        # reshape to add a sequence dimension of length 1
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
        
        # Apply transformer layers 
        x = self.transformer_encoder(x)
        
        # Pool and reshape
        x = x.transpose(1, 2)  # Shape: [batch_size, hidden_dim, 1]
        x = self.pooling(x).squeeze(-1)  # Shape: [batch_size, hidden_dim]
        
        # Apply FFN
        x = x + self.ffn(x)  # Residual connection
        
        # Output predictions
        y_1d = self.output_1d(x)
        y_30d = self.output_30d(x)
        
        return y_1d.squeeze(-1), y_30d.squeeze(-1)
    

def train_financial_model_cv(
    X, 
    y_1d, 
    y_30d, 
    n_splits=5,  # Number of folds for cross-validation
    batch_size=16, 
    num_epochs=50,
    learning_rate=0.001,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    weight_decay=1e-5,
    early_stopping_patience=5,
    device=None,
    return_best_model=True,  # Whether to return the best model across all folds
    random_state=42
):
    """
    Train a transformer-based model to predict financial returns from text embeddings
    using k-fold cross-validation.
    
    Args:
        X (numpy.ndarray): Text embeddings of shape [num_samples, embedding_dim]
        y_1d (numpy.ndarray): 1-day returns of shape [num_samples]
        y_30d (numpy.ndarray): 30-day returns of shape [num_samples]
        n_splits (int): Number of folds for cross-validation
        batch_size (int): Batch size for training
        num_epochs (int): Maximum number of training epochs
        learning_rate (float): Learning rate for Adam optimizer
        hidden_dim (int): Hidden dimension size for the transformer
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        weight_decay (float): L2 regularization strength
        early_stopping_patience (int): Number of epochs to wait for improvement
        device (torch.device): Device to use for training (CPU/GPU)
        return_best_model (bool): If True, return the model with best validation performance across folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing training results for each fold and overall statistics
    """
    # Set device
    if device is None:
        device = get_optimal_device()
    print(f"Using device: {device}")
    
    # Convert data to numpy arrays if they're tensors
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y_1d, torch.Tensor):
        y_1d = y_1d.cpu().numpy()
    if isinstance(y_30d, torch.Tensor):
        y_30d = y_30d.cpu().numpy()
    
    # Initialize k-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    cv_results = {
        'fold_results': [],
        'val_loss_1d': [],
        'val_loss_30d': [],
        'best_model_state': None,
        'best_val_loss_1d': float('inf'),
        'best_val_loss_30d': float('inf'),
        'best_fold': -1
    }
    
    # Keep track of overall best model across all folds
    best_model_state = None
    best_val_loss_1d = float('inf')
    best_fold = -1
    
    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_1d_train, y_1d_val = y_1d[train_idx], y_1d[val_idx]
        y_30d_train, y_30d_val = y_30d[train_idx], y_30d[val_idx]
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_1d_train_tensor = torch.tensor(y_1d_train, dtype=torch.float32)
        y_30d_train_tensor = torch.tensor(y_30d_train, dtype=torch.float32)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_1d_val_tensor = torch.tensor(y_1d_val, dtype=torch.float32)
        y_30d_val_tensor = torch.tensor(y_30d_val, dtype=torch.float32)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_1d_train_tensor, y_30d_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_1d_val_tensor, y_30d_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = X.shape[1]  # Embedding dimension
        model = FinancialTransformer(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - using 1D loss for scheduling
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # Initialize variables for training
        best_fold_val_loss_1d = float('inf')
        best_fold_val_loss_30d = float('inf')
        best_fold_model_state = None
        patience_counter = 0
        history = {
            'train_loss_1d': [],
            'train_loss_30d': [],
            'val_loss_1d': [],
            'val_loss_30d': []
        }
        
        # Training loop
        print("Starting training for this fold...")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss_1d = 0.0
            train_loss_30d = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for X_batch, y_1d_batch, y_30d_batch in progress_bar:
                X_batch = X_batch.to(device)
                y_1d_batch = y_1d_batch.to(device)
                y_30d_batch = y_30d_batch.to(device)
                
                # Forward pass
                y_1d_pred, y_30d_pred = model(X_batch)
                
                # Calculate losses for each target
                loss_1d = criterion(y_1d_pred, y_1d_batch)
                loss_30d = criterion(y_30d_pred, y_30d_batch)
                
                # Calculate gradients
                optimizer.zero_grad()
                total_loss = loss_1d + loss_30d
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                train_loss_1d += loss_1d.item()
                train_loss_30d += loss_30d.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss_1d': loss_1d.item(),
                    'loss_30d': loss_30d.item()
                })
            
            # Calculate average training metrics
            train_loss_1d /= len(train_loader)
            train_loss_30d /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss_1d = 0.0
            val_loss_30d = 0.0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for X_batch, y_1d_batch, y_30d_batch in progress_bar:
                    X_batch = X_batch.to(device)
                    y_1d_batch = y_1d_batch.to(device)
                    y_30d_batch = y_30d_batch.to(device)
                    
                    # Forward pass
                    y_1d_pred, y_30d_pred = model(X_batch)
                    
                    # Calculate losses for each target
                    loss_1d = criterion(y_1d_pred, y_1d_batch)
                    loss_30d = criterion(y_30d_pred, y_30d_batch)
                    
                    # Update metrics
                    val_loss_1d += loss_1d.item()
                    val_loss_30d += loss_30d.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss_1d': loss_1d.item(),
                        'loss_30d': loss_30d.item()
                    })
            
            # Calculate average validation metrics
            val_loss_1d /= len(val_loader)
            val_loss_30d /= len(val_loader)
            
            # Update learning rate based on 1-day validation loss
            scheduler.step(val_loss_1d)
            
            # Save metrics
            history['train_loss_1d'].append(train_loss_1d)
            history['train_loss_30d'].append(train_loss_30d)
            history['val_loss_1d'].append(val_loss_1d)
            history['val_loss_30d'].append(val_loss_30d)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: 1d: {train_loss_1d:.6f}, 30d: {train_loss_30d:.6f} - "
                f"Val Loss: 1d: {val_loss_1d:.6f}, 30d: {val_loss_30d:.6f}")
            
            # Check for improvement in 1-day prediction (primary focus)
            if val_loss_1d < best_fold_val_loss_1d:
                best_fold_val_loss_1d = val_loss_1d
                best_fold_val_loss_30d = val_loss_30d  # Record corresponding 30d loss
                best_fold_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"New best model for fold {fold+1} with 1-day validation loss: {best_fold_val_loss_1d:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'val_loss_1d': best_fold_val_loss_1d,
            'val_loss_30d': best_fold_val_loss_30d,
            'history': history,
            'model_state': best_fold_model_state
        }
        cv_results['fold_results'].append(fold_result)
        cv_results['val_loss_1d'].append(best_fold_val_loss_1d)
        cv_results['val_loss_30d'].append(best_fold_val_loss_30d)
                
                # Plot fold training history
        plt.figure(figsize=(15, 5))

        # Plot 1-day return loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss_1d'], label='Train Loss')
        plt.plot(history['val_loss_1d'], label='Validation Loss')
        plt.title(f'Fold {fold+1}: 1-Day Return Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot 30-day return loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss_30d'], label='Train Loss')
        plt.plot(history['val_loss_30d'], label='Validation Loss')
        plt.title(f'Fold {fold+1}: 30-Day Return Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'training_history_fold_{fold+1}.png')
        plt.close()



        
        
        # Update best model across all folds
        if best_fold_val_loss_1d < best_val_loss_1d:
            best_val_loss_1d = best_fold_val_loss_1d
            best_model_state = best_fold_model_state
            best_fold = fold + 1
            print(f"New best model across all folds: Fold {best_fold} with 1-day validation loss: {best_val_loss_1d:.6f}")
    
    # Calculate average performance across folds
    avg_val_loss_1d = np.mean(cv_results['val_loss_1d'])
    avg_val_loss_30d = np.mean(cv_results['val_loss_30d'])
    std_val_loss_1d = np.std(cv_results['val_loss_1d'])
    std_val_loss_30d = np.std(cv_results['val_loss_30d'])
    
    print("\n" + "="*50)
    print("Cross-Validation Results:")
    print(f"Average 1-day validation loss: {avg_val_loss_1d:.6f} ± {std_val_loss_1d:.6f}")
    print(f"Average 30-day validation loss: {avg_val_loss_30d:.6f} ± {std_val_loss_30d:.6f}")
    print(f"Best fold: {best_fold} with 1-day validation loss: {best_val_loss_1d:.6f}")
    print("="*50)
    
    # Update the cv_results dictionary with final statistics
    cv_results['avg_val_loss_1d'] = avg_val_loss_1d
    cv_results['avg_val_loss_30d'] = avg_val_loss_30d
    cv_results['std_val_loss_1d'] = std_val_loss_1d
    cv_results['std_val_loss_30d'] = std_val_loss_30d
    cv_results['best_val_loss_1d'] = best_val_loss_1d
    cv_results['best_val_loss_30d'] = best_fold_val_loss_30d
    cv_results['best_fold'] = best_fold
    cv_results['best_model_state'] = best_model_state
    
    # Initialize and return the best model if requested
    if return_best_model:
        print(f"Initializing the best model from fold {best_fold}")
        best_model = FinancialTransformer(
            input_dim=X.shape[1], 
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        best_model.load_state_dict(best_model_state)
        cv_results['model'] = best_model
    
    return cv_results

def hyperparameter_tuning(
    X, 
    y_1d, 
    y_30d, 
    n_trials=5,  # Number of hyperparameter combinations to try
    n_splits=3,    # Number of cross-validation folds for each trial
    device=None,
    random_state=42
):
    """
    Perform hyperparameter tuning for the financial transformer model using Optuna.
    
    Args:
        X (numpy.ndarray): Input features
        y_1d (numpy.ndarray): 1-day returns
        y_30d (numpy.ndarray): 30-day returns
        n_trials (int): Number of hyperparameter combinations to try
        n_splits (int): Number of cross-validation folds for each trial
        device (torch.device): Device to use for training
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Best hyperparameters and evaluation results
    """
    import optuna
    from optuna.samplers import TPESampler
    
    # Set device
    if device is None:
        device = get_optimal_device()
    print(f"Using device: {device}")
    
    # Convert data to numpy arrays if they're tensors
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y_1d, torch.Tensor):
        y_1d = y_1d.cpu().numpy()
    if isinstance(y_30d, torch.Tensor):
        y_30d = y_30d.cpu().numpy()
    
    # Calculate the number of samples
    n_samples = len(X)
    
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters to tune
        hyperparams = {
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
        
        # Ensure batch_size is not larger than the dataset
        hyperparams['batch_size'] = min(hyperparams['batch_size'], n_samples)
        
        # Ensure num_heads divides hidden_dim evenly (required for transformer architecture)
        while hyperparams['hidden_dim'] % hyperparams['num_heads'] != 0:
            hyperparams['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
        
        # Set fixed parameters
        fixed_params = {
            'num_epochs': 30,  # Limit epochs for faster tuning
            'early_stopping_patience': 5,
            'return_best_model': False,  # Don't need the model object, just metrics
            'random_state': random_state,
            'device': device
        }
        
        # Combine all parameters
        params = {**hyperparams, **fixed_params}
        
        try:
            # Run cross-validation with these parameters
            cv_results = train_financial_model_cv(
                X=X,
                y_1d=y_1d,
                y_30d=y_30d,
                n_splits=n_splits,
                **params
            )
            
            # Use the average 1-day validation loss as the objective to minimize
            avg_val_loss_1d = cv_results['avg_val_loss_1d']
            
            # Log additional metrics for analysis
            trial.set_user_attr('avg_val_loss_30d', cv_results['avg_val_loss_30d'])
            trial.set_user_attr('std_val_loss_1d', cv_results['std_val_loss_1d'])
            trial.set_user_attr('best_fold', cv_results['best_fold'])
            trial.set_user_attr('best_val_loss_1d', cv_results['best_val_loss_1d'])
            
            return avg_val_loss_1d
            
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            # Return a high loss value if the trial fails
            return float('inf')
    
    # Create Optuna study with TPE sampler (good for hyperparameter optimization)
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Run optimization
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*50)
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best 1-day validation loss: {best_value:.6f}")
    print("="*50)
    
    # Train final model with best parameters
    print("\nTraining final model with best hyperparameters...")
    final_params = {
        **best_params,
        'num_epochs': 50,  # Use more epochs for final model
        'early_stopping_patience': 5,
        'return_best_model': True,
        'random_state': random_state,
        'device': device
    }
    
    final_results = train_financial_model_cv(
        X=X,
        y_1d=y_1d,
        y_30d=y_30d,
        n_splits=n_splits,
        **final_params
    )
        
        # Visualize hyperparameter importance
    try:
        plt.figure(figsize=(10, 6))
        importance = optuna.importance.get_param_importances(study)

        params = list(importance.keys())
        scores = list(importance.values())

        indices = np.arange(len(params))

        plt.barh(indices, scores, align='center')
        plt.yticks(indices, params)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig('hyperparameter_importance.png')
        plt.close()
    except Exception as e:
        print(f"Could not visualize hyperparameter importance: {str(e)}")

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig('optimization_history.png')
    plt.close()

    # Visualize parameter relationships if we have more than 10 trials
    if n_trials > 10:
        plt.figure(figsize=(12, 10))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('parameter_importances.png')
        plt.close()

        # Plot slice plot for most important parameters
        important_params = list(importance.keys())[:3]  # Top 3 parameters
        for param in important_params:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_slice(study, params=[param])
            plt.tight_layout()
            plt.savefig(f'slice_plot_{param}.png')
            plt.close()

    # Return results
    return {
        'best_hyperparams': best_params,
        'best_validation_loss': best_value,
        'study': study,
        'final_model': final_results['model'],
        'final_model_performance': {
            'avg_val_loss_1d': final_results['avg_val_loss_1d'],
            'avg_val_loss_30d': final_results['avg_val_loss_30d'],
            'best_fold': final_results['best_fold'],
            'best_val_loss_1d': final_results['best_val_loss_1d'],
        }
    }

def train_with_hyperparameter_tuning(X, y_1d, y_30d, n_trials=20, n_splits=3):
    """
    Complete pipeline for model training with hyperparameter tuning.
    
    Args:
        X (numpy.ndarray): Input features
        y_1d (numpy.ndarray): 1-day returns
        y_30d (numpy.ndarray): 30-day returns
        n_trials (int): Number of hyperparameter combinations to try
        n_splits (int): Number of cross-validation folds for each trial
        
    Returns:
        dict: Training results including the best model
    """
    # Perform hyperparameter tuning
    tuning_results = hyperparameter_tuning(
        X=X,
        y_1d=y_1d,
        y_30d=y_30d,
        n_trials=n_trials,
        n_splits=n_splits
    )
    
    # Get the best model
    model = tuning_results['final_model']
    
    # Make predictions with the best model
    y_1d_pred, y_30d_pred = predict_returns(model, X)
    
    # Print final performance summary
    print("\n" + "="*50)
    print("Final Model Performance:")
    print(f"Average 1-day validation loss: {tuning_results['final_model_performance']['avg_val_loss_1d']:.6f}")
    print(f"Average 30-day validation loss: {tuning_results['final_model_performance']['avg_val_loss_30d']:.6f}")
    print(f"Best fold: {tuning_results['final_model_performance']['best_fold']}")
    print(f"Best 1-day validation loss: {tuning_results['final_model_performance']['best_val_loss_1d']:.6f}")
    print("="*50)
    
    # Save the best model
    torch.save(model.state_dict(), "best_financial_transformer_model.pt")
    print("\nBest model saved to 'best_financial_transformer_model.pt'")
    
    # Save the hyperparameters for future reference
    import json
    with open("best_hyperparameters.json", "w") as f:
        json.dump(tuning_results['best_hyperparams'], f, indent=4)
    print("Best hyperparameters saved to 'best_hyperparameters.json'")
    
    return {
        'model': model,
        'predictions': {
            'y_1d_pred': y_1d_pred,
            'y_30d_pred': y_30d_pred
        },
        'hyperparameters': tuning_results['best_hyperparams'],
        'performance': tuning_results['final_model_performance']
    }


# Function to make predictions with the trained model
def predict_returns(model, X, device=None):
    """
    Make predictions with the trained model.
    
    Args:
        model (FinancialTransformer): Trained model
        X (numpy.ndarray): Input features
        device (torch.device): Device to use
        
    Returns:
        tuple: Predicted 1-day and 30-day returns
    """
    if device is None:
        device = get_optimal_device()
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_1d_pred, y_30d_pred = model(X_tensor)
    
    # Convert to numpy arrays
    y_1d_pred = y_1d_pred.cpu().numpy()
    y_30d_pred = y_30d_pred.cpu().numpy()
    
    return y_1d_pred, y_30d_pred

# === DATA AGGREGATION AND FACTOR INCLUSION === #

def aggregate_weighted_predictions(predictions, weights):
    """
    Aggregate predictions using market value weights.
    
    Args:
        predictions (numpy.ndarray): Predicted returns array
        weights (numpy.ndarray): Market value weights array
        
    Returns:
        float: Weighted average prediction
    """
    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()
    
    # Calculate weighted average
    weighted_prediction = (predictions * normalized_weights).sum()
    
    return weighted_prediction

def aggregate_predictions_by_date(sampled_data, y_1d_pred, y_30d_pred):
    """
    Aggregate predictions by date using market value weights.
    
    Args:
        sampled_data (dict): Dictionary containing the financial data with dates
        y_1d_pred (numpy.ndarray): Predicted 1-day returns
        y_30d_pred (numpy.ndarray): Predicted 30-day returns
        
    Returns:
        dict: Dictionary of date-based aggregated predictions
    """
    # Create a dictionary to store date-based data
    date_data = {}
    
    # Group data by date
    for i, (filename, entry) in enumerate(sampled_data.items()):
        date = entry['date']
        
        if date not in date_data:
            date_data[date] = {
                'predictions_1d': [],
                'predictions_30d': [],
                'market_values': [],
                'indices': []
            }
        
        date_data[date]['predictions_1d'].append(y_1d_pred[i])
        date_data[date]['predictions_30d'].append(y_30d_pred[i])
        date_data[date]['market_values'].append(entry['MarketValue'])
        date_data[date]['indices'].append(i)
    
    # Calculate weighted predictions for each date
    results = {}
    for date, data in date_data.items():
        # Convert lists to numpy arrays
        pred_1d = np.array(data['predictions_1d'])
        pred_30d = np.array(data['predictions_30d'])
        market_values = np.array(data['market_values'])
        
        # Calculate weighted predictions
        weighted_1d = aggregate_weighted_predictions(pred_1d, market_values)
        weighted_30d = aggregate_weighted_predictions(pred_30d, market_values)
        
        results[date] = {
            'weighted_1d_pred': weighted_1d,
            'weighted_30d_pred': weighted_30d,
            'sample_count': len(data['indices']),
            'total_market_value': sum(market_values)
        }
    
    return results

def load_fama_french_factors(filepath):
    """
    Load Fama-French factors from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with datetime.date objects as keys and factor values as values
    """
    
    # Load the CSV file
    try:
        df = pd.read_csv(filepath)
        if 'dateff' in df.columns:
            df.rename(columns={'dateff': 'date'}, inplace=True)
        
        # Ensure date column exists
        if 'date' not in df.columns:
            print(f"Warning: 'date' column not found in {filepath}")
            print(f"Available columns: {df.columns.tolist()}")
            return {}
        
        # Convert to dictionary with date object as key
        factors_dict = {}
        for _, row in df.iterrows():
            date_str = str(row['date'])
            
            # Convert string to date object based on format
            if len(date_str) == 8:  # YYYYMMDD format
                date_obj = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            elif len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD format
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                print(f"Warning: Unrecognized date format: {date_str}")
                continue
            
            # Create dictionary of factors for this date
            factors = {col: row[col] for col in df.columns if col != 'date'}
            factors_dict[date_obj] = factors
            
        return factors_dict
        
    except Exception as e:
        print(f"Error loading Fama-French factors: {str(e)}")
        return {}

def merge_predictions_with_factors(date_predictions, ff_factors, frequency = 'daily'):
    """
    Merge predictions with Fama-French factors based on the given frequency.

    Args:
        date_predictions (dict): Dictionary of date-based predictions.
        ff_factors (dict): Dictionary of Fama-French factors by date.

    Returns:
        dict: Merged dictionary with predictions and factors.
    """
    merged_data = {}

    # Track matching and non-matching dates
    matched_dates = []
    prediction_only_dates = []
    factor_only_dates = []

    # Helper function for date formatting
    def format_date(d):
        if isinstance(d, str):
            d = datetime.datetime.strptime(d, "%Y-%m-%d")
        elif isinstance(d, datetime.date):
            d = datetime.datetime.combine(d, datetime.datetime.min.time())
        return d.strftime("%Y-%m") if frequency == 'monthly' else d.strftime("%Y-%m-%d")

    # Format predictions and factors keys
    formatted_predictions = {format_date(date): data for date, data in date_predictions.items()}
    formatted_factors = {format_date(date): data for date, data in ff_factors.items()}

    # Merge based on formatted keys
    all_dates = set(formatted_predictions) | set(formatted_factors)

    for date in all_dates:
        pred_data = formatted_predictions.get(date)
        factor_data = formatted_factors.get(date)

        if pred_data and factor_data:
            merged_data[date] = {**pred_data, **factor_data}
            matched_dates.append(date)
        elif pred_data:
            merged_data[date] = pred_data
            prediction_only_dates.append(date)
        elif factor_data:
            merged_data[date] = factor_data
            factor_only_dates.append(date)

    # Print matching statistics
    print(f"\nDate Matching Statistics:")
    print(f"  Total prediction dates: {len(formatted_predictions)}")
    print(f"  Total factor dates: {len(formatted_factors)}")
    print(f"  Dates with both predictions and factors: {len(matched_dates)}")
    print(f"  Dates with predictions only: {len(prediction_only_dates)}")
    print(f"  Dates with factors only: {len(factor_only_dates)}")

    return merged_data

# === REGRESSION VS FAMA-FREMCH FACTORS === #

def run_fama_french_regression(merged_df, factorsFrequency = None):
    """
    Run OLS regression with y_1d_pred as dependent variable and Fama-French factors as independent variables.
    Output is formatted similar to Stata.
    
    Args:
        merged_df (pd.DataFrame): DataFrame containing predictions and Fama-French factors
        
    Returns:
        None (prints regression results)
    """

    # Make a copy to avoid modifying the original
    df = merged_df.copy()
    
    # Standard Fama-French factors to include if available
    ff_factors = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd', 'rf']
    
    # Filter to only include factors that exist in the dataframe
    available_factors = [f for f in ff_factors if f in df.columns]
        
    # Drop rows with missing values in dependent variable or factors
    df = df.dropna(subset=['weighted_1d_pred'] + available_factors)
    
    # Prepare X and y for regression
    X = df[available_factors]
    if factorsFrequency == 'daily':
        y = df['weighted_1d_pred']
    elif factorsFrequency == 'monthly':
        y = df['weighted_30d_pred']
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Print results in a format similar to Stata
    print("\n" + "="*80)
    print("OLS Regression Results (Stata-style)")
    print("="*80)
    print(results.summary())
        
    # --- Save coefficients table ---
    coef_table = results.summary2().tables[1]
    coef_table.to_csv("ols_coefficients.csv")

    # --- Save model-level statistics ---
    model_stats = {
        "Dep. Variable": results.model.endog_names,
        "R-squared": results.rsquared,
        "Adj. R-squared": results.rsquared_adj,
        "F-statistic": results.fvalue,
        "Prob (F-statistic)": results.f_pvalue,
        "Log-Likelihood": results.llf,
        "AIC": results.aic,
        "BIC": results.bic,
        "No. Observations": int(results.nobs),
        "Df Residuals": int(results.df_resid),
        "Df Model": int(results.df_model),
        "Covariance Type": results.cov_type,
    }

    # Convert to DataFrame and save
    model_stats_df = pd.DataFrame([model_stats])
    model_stats_df.to_csv("ols_model_summary.csv", index=False)
        
    return results

# === MAIN EXECUTION === #

if __name__ == "__main__":

        # Config
    base_dir = os.getcwd()
    folder_path = os.path.join(base_dir, 'eight_k_htm')
    ff_daily_factors_path = os.path.join(os.path.dirname(folder_path), 'ff_daily_factors.csv')
    ff_monthly_factors_path = os.path.join(os.path.dirname(folder_path), 'ff_monthly_factors.csv')
    sp500_data_path = os.path.join(os.path.dirname(folder_path), 'sp500_data.csv')
    N = 56000  # Sample size or use None for all files
    embedding_dim_model = 'finbert'  # 'finbert' or 'ollama' or 'bert'

    # Load SP500 data
    print(f"Loading SP500 data from: {sp500_data_path}")
    sp500_data = load_sp500_data(sp500_data_path)
    
    # Sample and extract text from HTM files with SP500 data
    sampled_data = sample_htm_files(folder_path, sp500_data, N)

    # Filter out entries with missing return values
    filtered_data = filter_missing_returns(sampled_data)

    # Extract text from HTM files
    texts = [entry['text'] for entry in filtered_data.values()]
    X = vectorize_text(texts, model_type=embedding_dim_model)
    y_1d = np.array([entry['1D_return'] for entry in filtered_data.values()])
    y_30d = np.array([entry['30D_return'] for entry in filtered_data.values()])
    
    # Extract market values for weighting
    market_values = np.array([entry['MarketValue'] for entry in filtered_data.values()])

    # Check if we have enough samples
    if len(y_1d) >= 10:  # Minimum requirement for a tiny model

        # Train with hyperparameter tuning (slower but more optimal)
        results = train_with_hyperparameter_tuning(
            X=X,
            y_1d=y_1d,
            y_30d=y_30d,
            n_trials=5,  # Adjust based on your computational budget
            n_splits=3    # Fewer splits for faster tuning
        )
        
        # Get the best model and predictions
        model = results['model']
        y_1d_pred = results['predictions']['y_1d_pred']
        y_30d_pred = results['predictions']['y_30d_pred']
        
        # Calculate overall market value-weighted aggregate predictions
        weighted_1d_pred = aggregate_weighted_predictions(y_1d_pred, market_values)
        weighted_30d_pred = aggregate_weighted_predictions(y_30d_pred, market_values)
        
        # Calculate date-based weighted predictions
        date_predictions = aggregate_predictions_by_date(filtered_data, y_1d_pred, y_30d_pred)
        
        # Load Fama-French factors
        print(f"\nLoading Fama-French factors from: {ff_daily_factors_path}")
            # Daily factors
        ff_daily_factors = load_fama_french_factors(ff_daily_factors_path)
            # Monthly factors
        ff_monthly_factors = load_fama_french_factors(ff_monthly_factors_path)
        print(f"Loaded factors for {len(ff_daily_factors)} dates")

        for name, ff_factors in [('daily', ff_daily_factors), ('monthly', ff_monthly_factors)]:

            # Merge predictions with factors
            merged_data = merge_predictions_with_factors(date_predictions, ff_factors, frequency=name)
            
            # Sort dates chronologically
            sorted_dates = sorted(merged_data.keys())
            
            # Save the model
            torch.save(model.state_dict(), "financial_transformer_model.pt")
            print("\nModel saved to 'financial_transformer_model.pt'")
            
            # Convert merged data to DataFrame
            merged_df = pd.DataFrame()
            for date, data in merged_data.items():
                row = {'date': date}
                row.update(data)
                merged_df = pd.concat([merged_df, pd.DataFrame([row])], ignore_index=True)
            
            # Sort by date
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df = merged_df.sort_values('date')
            
            print("\nAnalysis complete!")

            # Run the regression analysis
            regression_results = run_fama_french_regression(merged_df, factorsFrequency = name)
