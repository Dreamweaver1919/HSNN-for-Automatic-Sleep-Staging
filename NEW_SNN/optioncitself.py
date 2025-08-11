"""
Complete Standalone Test Script for Option C: CNN-Reservoir Models
================================================================
Comprehensive testing with detailed visualizations and robust error handling.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                           precision_recall_fscore_support, accuracy_score)
from spikingjelly.activation_based import neuron, surrogate
import tqdm
from datetime import datetime
import logging
import warnings
import mne
import pandas as pd
from collections import Counter
import json

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# Suppress warnings
warnings.filterwarnings('ignore', message='Precision loss occurred')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
mne.set_log_level('WARNING')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================
# Data Loading and Preprocessing
# ================================

class EEGDataset(torch.utils.data.Dataset):
    """Dataset class for EEG data with proper dtype handling."""
    def __init__(self, trials):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        segment, label = self.trials[idx]
        # Ensure proper dtype - this fixes the double/float error
        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return segment_tensor, label_tensor


def load_and_preprocess_data(edf_file, mat_file, use_mne_cleaning=True):
    """
    Load EEG data and sleep stages with proper dtype handling.
    """
    
    logger.info("="*70)
    logger.info("LOADING AND PREPROCESSING EEG DATA FOR OPTION C")
    logger.info("="*70)
    
    # Load EEG data using MNE
    logger.info(f"Loading EEG data from: {os.path.basename(edf_file)}")
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        # Remove non-EEG channels
        non_eeg_patterns = ['Event', 'ECG', 'EOG', 'EMG', 'Resp', 'Status', 'Marker']
        channels_to_drop = []
        for ch_name in raw.ch_names:
            for pattern in non_eeg_patterns:
                if pattern.lower() in ch_name.lower():
                    channels_to_drop.append(ch_name)
                    break
        
        if channels_to_drop:
            logger.info(f"Removing non-EEG channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)
        
        logger.info(f"EEG channels: {len(raw.ch_names)} channels")
        logger.info(f"Channel names: {raw.ch_names}")
        logger.info(f"Sampling frequency: {sfreq} Hz")
        logger.info(f"Duration: {raw.times[-1]:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error reading EDF file: {e}")
        raise

    # Apply MNE preprocessing if requested
    if use_mne_cleaning and len(raw.ch_names) > 0:
        logger.info("Applying basic preprocessing...")
        try:
            # Apply bandpass filter
            raw.filter(0.5, 45, fir_design='firwin', verbose=False)
            logger.info("Applied bandpass filter (0.5-45 Hz)")
        except Exception as e:
            logger.warning(f"Could not apply filter: {e}")
    
    # Get preprocessed data
    eeg_data = raw.get_data()
    n_channels = eeg_data.shape[0]
    
    # Load sleep staging information
    logger.info(f"Loading sleep stages from: {os.path.basename(mat_file)}")
    try:
        mat_contents = loadmat(mat_file)
        keys = [key for key in mat_contents.keys() if not key.startswith('__')]
        
        stages_raw = mat_contents[keys[0]]
        stages_raw = np.squeeze(stages_raw)
        stages = [str(s) for s in stages_raw]
        stages_clean = [s[0] if isinstance(s, (np.ndarray, list)) else s for s in stages]
        
        # Map stages to integers
        stage_mapping = {
            "['NREM 1']": 0, 
            "['NREM 2']": 1, 
            "['NREM 3']": 2, 
            "['Wake']": 3, 
            "['REM']": 4
        }
        stages_int = [stage_mapping.get(s, 3) for s in stages_clean]
        n_classes = len(set(stages_int))
        
        logger.info(f"Loaded {len(stages_int)} sleep stage labels")
        logger.info(f"Number of classes detected: {n_classes}")
        
        # Print stage distribution
        unique, counts = np.unique(stages_int, return_counts=True)
        stage_names = ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
        logger.info("Stage distribution:")
        for stage_id, count in zip(unique, counts):
            percentage = count/len(stages_int)*100
            logger.info(f"  {stage_names[stage_id]} (Class {stage_id}): {count} epochs ({percentage:.1f}%)")
            
    except Exception as e:
        logger.error(f"Error loading sleep stages: {e}")
        raise
    
    # Create epochs (30-second windows)
    logger.info("Creating 30-second epochs...")
    epoch_duration = 30
    samples_per_epoch = int(sfreq * epoch_duration)
    total_samples = eeg_data.shape[1]
    n_epochs_available = total_samples // samples_per_epoch
    n_epochs = min(len(stages_int), n_epochs_available)
    
    trials = []
    valid_epochs = 0
    skipped_epochs = 0
    
    for epoch_idx in range(n_epochs):
        start_idx = epoch_idx * samples_per_epoch
        end_idx = (epoch_idx + 1) * samples_per_epoch
        segment = eeg_data[:, start_idx:end_idx]
        
        # Check for valid data
        if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
            skipped_epochs += 1
            continue
        
        segment_std = np.std(segment)
        if segment_std < 1e-6 or segment_std > 1e4:
            skipped_epochs += 1
            continue
        
        stage = stages_int[epoch_idx]
        
        # CRITICAL: Ensure float32 to prevent dtype errors
        segment = segment.astype(np.float32)
        trials.append((segment, stage))
        valid_epochs += 1
    
    logger.info(f"Created {valid_epochs} valid epochs")
    if skipped_epochs > 0:
        logger.info(f"Skipped {skipped_epochs} epochs due to quality issues")
    
    # Create dataset
    dataset = EEGDataset(trials)
    
    # Compute normalization statistics with proper dtype
    all_segments = np.stack([seg for seg, _ in trials], axis=0)
    global_mean = np.mean(all_segments, axis=(0, 2)).astype(np.float32)
    global_std = np.std(all_segments, axis=(0, 2)).astype(np.float32)
    
    # Prevent division by zero
    global_std = np.where(global_std < 1e-6, 1.0, global_std).astype(np.float32)
    
    logger.info(f"Global mean shape: {global_mean.shape}")
    logger.info(f"Global std shape: {global_std.shape}")
    logger.info(f"Mean values: {global_mean}")
    logger.info(f"Std values: {global_std}")
    
    return dataset, n_channels, n_classes, sfreq, global_mean, global_std


def debug_data_classes(dataset, train_indices, eval_indices):
    """Debug function to check class distribution and identify potential issues."""
    logger.info("="*50)
    logger.info("DEBUGGING CLASS DISTRIBUTION")
    logger.info("="*50)
    
    # Check all data
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    unique_all, counts_all = np.unique(all_labels, return_counts=True)
    
    logger.info(f"All data - Total samples: {len(all_labels)}")
    logger.info(f"All data - Unique classes: {unique_all}")
    logger.info(f"All data - Class counts: {counts_all}")
    
    # Check training data
    train_labels = [dataset[i][1] for i in train_indices]
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    
    logger.info(f"Train data - Total samples: {len(train_labels)}")
    logger.info(f"Train data - Unique classes: {unique_train}")
    logger.info(f"Train data - Class counts: {counts_train}")
    
    # Check eval data
    eval_labels = [dataset[i][1] for i in eval_indices]
    unique_eval, counts_eval = np.unique(eval_labels, return_counts=True)
    
    logger.info(f"Eval data - Total samples: {len(eval_labels)}")
    logger.info(f"Eval data - Unique classes: {unique_eval}")
    logger.info(f"Eval data - Class counts: {counts_eval}")
    
    # Check for missing classes
    expected_classes = set(range(5))  # 0, 1, 2, 3, 4
    actual_train_classes = set(unique_train)
    actual_eval_classes = set(unique_eval)
    actual_all_classes = set(unique_all)
    
    missing_train = expected_classes - actual_train_classes
    missing_eval = expected_classes - actual_eval_classes
    missing_all = expected_classes - actual_all_classes
    
    if missing_train:
        logger.warning(f"Missing classes in training set: {missing_train}")
    if missing_eval:
        logger.warning(f"Missing classes in evaluation set: {missing_eval}")
    if missing_all:
        logger.warning(f"Missing classes in entire dataset: {missing_all}")
    
    logger.info("="*50)
    
    return {
        'n_classes_total': len(unique_all),
        'n_classes_train': len(unique_train),
        'n_classes_eval': len(unique_eval),
        'all_classes': list(unique_all),
        'train_classes': list(unique_train),
        'eval_classes': list(unique_eval),
        'missing_train': list(missing_train),
        'missing_eval': list(missing_eval),
        'class_counts_all': dict(zip(unique_all, counts_all)),
        'class_counts_train': dict(zip(unique_train, counts_train)),
        'class_counts_eval': dict(zip(unique_eval, counts_eval))
    }


def create_stratified_split(dataset, test_size=0.2, random_state=42):
    """Create stratified split to ensure balanced sleep stages."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    # Check if we have enough samples for stratification
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    
    if min_count < 2:
        logger.warning(f"Some classes have very few samples (min: {min_count}). Using random split instead.")
        # Fallback to random split
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        split_point = int(len(indices) * (1 - test_size))
        train_idx = shuffled_indices[:split_point]
        eval_idx = shuffled_indices[split_point:]
    else:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, 
                                            random_state=random_state)
            train_idx, eval_idx = next(splitter.split(indices, labels))
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            np.random.seed(random_state)
            shuffled_indices = np.random.permutation(indices)
            split_point = int(len(indices) * (1 - test_size))
            train_idx = shuffled_indices[:split_point]
            eval_idx = shuffled_indices[split_point:]
    
    # Print distribution
    train_labels = [labels[i] for i in train_idx]
    eval_labels = [labels[i] for i in eval_idx]
    
    stage_names = ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
    
    logger.info("Training set distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for stage, count in zip(unique, counts):
        if stage < len(stage_names):
            logger.info(f"  {stage_names[stage]} (Class {stage}): {count} ({count/len(train_labels)*100:.1f}%)")
    
    logger.info("Evaluation set distribution:")
    unique, counts = np.unique(eval_labels, return_counts=True)
    for stage, count in zip(unique, counts):
        if stage < len(stage_names):
            logger.info(f"  {stage_names[stage]} (Class {stage}): {count} ({count/len(eval_labels)*100:.1f}%)")
    
    return train_idx.tolist(), eval_idx.tolist()


# ================================
# Model Definitions (from Option C)
# ================================

class SpikingJellyReservoirSNN(nn.Module):
    """Fixed-weight reservoir using SpikingJelly LIF neurons."""
    
    def __init__(self, n_channels, n_reservoir, sfreq, tau=0.02, threshold=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_reservoir = n_reservoir
        self.sfreq = sfreq
        self.tau = tau
        self.threshold = threshold

        # Fully-connected layers
        self.fc_in = nn.Linear(n_channels, n_reservoir)
        self.fc_rec = nn.Linear(n_reservoir, n_reservoir)

        # Freeze weights (reservoir remains fixed)
        for param in self.fc_in.parameters():
            param.requires_grad = False
        for param in self.fc_rec.parameters():
            param.requires_grad = False

        # Initialize weights
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc_in.weight)
            nn.init.orthogonal_(self.fc_rec.weight)
            # Control spectral radius
            weight = self.fc_rec.weight.data
            abs_eigenvalues = torch.abs(torch.linalg.eigvals(weight))
            spectral_radius = abs_eigenvalues.max()
            self.fc_rec.weight.data = (weight / spectral_radius) * 0.9

        # Create LIF neuron
        tau_timesteps = tau * sfreq
        self.lif = neuron.LIFNode(tau=tau_timesteps, surrogate_function=surrogate.ATan(), 
                                 detach_reset=True)
        self.lif.v_threshold = threshold

    def forward(self, x):
        """Process input through reservoir."""
        batch_size, _, time_steps = x.shape
        device = x.device
        spike_counts = torch.zeros(batch_size, self.n_reservoir, device=device)
        spikes = torch.zeros(batch_size, self.n_reservoir, device=device)

        self.lif.reset()

        for t in range(time_steps):
            input_t = x[:, :, t]
            I_in = self.fc_in(input_t) + self.fc_rec(spikes)
            spikes = self.lif(I_in)
            spike_counts += spikes

        return spike_counts


class ComplexClassifier(nn.Module):
    """Multi-layer classifier for sleep stage classification."""
    
    def __init__(self, n_reservoir, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_reservoir, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class SimplifiedSNNModel(nn.Module):
    """Original CNN-Reservoir model."""
    
    def __init__(self, n_channels, n_reservoir, n_classes, sfreq, tau=0.02, threshold=1.0):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.reservoir = SpikingJellyReservoirSNN(64, n_reservoir, sfreq, tau, threshold)
        self.classifier = ComplexClassifier(n_reservoir, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        spike_counts = self.reservoir(x)
        rates = self.classifier(spike_counts)
        return rates


class EnhancedSNNModel(nn.Module):
    """Enhanced model with attention mechanism."""
    
    def __init__(self, n_channels, n_reservoir, n_classes, sfreq, tau=0.02, threshold=1.0):
        super().__init__()
        
        # Enhanced CNN with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Reservoir and classifier
        self.reservoir = SpikingJellyReservoirSNN(64, n_reservoir, sfreq, tau, threshold)
        self.classifier = ComplexClassifier(n_reservoir, n_classes)
        
    def forward(self, x):
        # CNN with skip connections
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        
        # Apply attention
        att_weights = self.attention(h3)
        h3_attended = h3 * att_weights
        
        # Reservoir processing
        spike_counts = self.reservoir(h3_attended)
        
        # Classification
        output = self.classifier(spike_counts)
        return output, att_weights


# ================================
# Training Pipeline
# ================================

class OptionCPipeline:
    """Training pipeline for Option C models."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def create_dataloader(self, dataset, indices, global_mean, global_std, 
                         batch_size, shuffle=True):
        """Create DataLoader with proper normalization and dtype handling."""
        
        class NormalizedDataset(Dataset):
            def __init__(self, dataset, indices, global_mean, global_std):
                self.dataset = dataset
                self.indices = indices
                # Ensure float32 for normalization parameters
                self.global_mean = global_mean.astype(np.float32)
                self.global_std = global_std.astype(np.float32)
                
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                segment, label = self.dataset[real_idx]
                
                # Ensure proper dtype
                if isinstance(segment, torch.Tensor):
                    segment = segment.float()
                else:
                    segment = torch.tensor(segment, dtype=torch.float32)
                
                # Normalize with float32 tensors
                mean_tensor = torch.tensor(self.global_mean[:, None], dtype=torch.float32)
                std_tensor = torch.tensor(self.global_std[:, None], dtype=torch.float32)
                segment = (segment - mean_tensor) / std_tensor
                
                # Ensure label is long tensor
                if isinstance(label, torch.Tensor):
                    label = label.long()
                else:
                    label = torch.tensor(label, dtype=torch.long)
                    
                return segment, label
        
        normalized_dataset = NormalizedDataset(dataset, indices, global_mean, global_std)
        return DataLoader(normalized_dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=0)  # num_workers=0 to avoid multiprocessing issues
    
    def train_model(self, model, train_loader, eval_loader, n_epochs, learning_rate, 
                   weight_decay, model_name, use_enhanced=False):
        """Train a single model."""
        
        logger.info(f"Training {model_name}...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        history = {'train_loss': [], 'train_acc': [], 'eval_loss': [], 'eval_acc': []}
        best_acc = 0
        patience = 0
        max_patience = 15
        
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for batch_idx, (segments, labels) in enumerate(pbar):
                segments = segments.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                if use_enhanced:
                    outputs, att_weights = model(segments)
                else:
                    outputs = model(segments)
                    
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
            scheduler.step()
            
            # Evaluation phase
            eval_loss, eval_acc = self.evaluate(model, eval_loader, criterion, use_enhanced)
            
            # Record history
            history['train_loss'].append(train_loss / train_total)
            history['train_acc'].append(100. * train_correct / train_total)
            history['eval_loss'].append(eval_loss)
            history['eval_acc'].append(eval_acc)
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {history["train_loss"][-1]:.4f}, '
                       f'Train Acc: {history["train_acc"][-1]:.2f}%, '
                       f'Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%')
            
            # Save best model
            if eval_acc > best_acc:
                best_acc = eval_acc
                model_filename = f'best_{model_name.lower().replace(" ", "_").replace("-", "_")}.pth'
                torch.save(model.state_dict(), model_filename)
                patience = 0
                logger.info(f'✓ New best model saved! (Eval Acc: {eval_acc:.2f}%)')
            else:
                patience += 1
                
            # Early stopping
            if patience >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Check for convergence
            if history['train_acc'][-1] >= 95 and eval_acc >= 90:
                logger.info("High accuracy achieved, stopping training")
                break
                
        return model, history, best_acc
    
    def evaluate(self, model, loader, criterion, use_enhanced=False):
        """Evaluate model performance."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for segments, labels in loader:
                segments = segments.to(self.device)
                labels = labels.to(self.device)
                
                if use_enhanced:
                    outputs, _ = model(segments)
                else:
                    outputs = model(segments)
                    
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / total, 100. * correct / total
    
    def evaluate_final(self, model, loader, use_enhanced=False):
        """Final detailed evaluation."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for segments, labels in tqdm.tqdm(loader, desc="Final evaluation"):
                segments = segments.to(self.device)
                
                if use_enhanced:
                    outputs, _ = model(segments)
                else:
                    outputs = model(segments)
                    
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'accuracy': 100. * np.mean(np.array(all_preds) == np.array(all_labels)),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }


# ================================
# Comprehensive Visualization
# ================================

def create_data_overview_plot(class_debug_info, stage_names):
    """Create overview plot of data distribution."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall class distribution
    classes = class_debug_info['all_classes']
    counts = [class_debug_info['class_counts_all'][c] for c in classes]
    class_labels = [stage_names[c] for c in classes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars1 = ax1.bar(class_labels, counts, color=colors)
    ax1.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Epochs')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Train vs Eval distribution
    train_counts = [class_debug_info['class_counts_train'].get(c, 0) for c in classes]
    eval_counts = [class_debug_info['class_counts_eval'].get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax2.bar(x - width/2, train_counts, width, label='Training', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, eval_counts, width, label='Evaluation', alpha=0.8, color='lightcoral')
    
    ax2.set_title('Train vs Eval Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Epochs')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, rotation=45)
    ax2.legend()
    
    # Pie chart of overall distribution
    ax3.pie(counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax3.set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    # Summary statistics
    ax4.axis('off')
    summary_text = f"""Dataset Summary Statistics

Total Epochs: {sum(counts)}
Number of Classes: {len(classes)}
Classes Present: {', '.join(class_labels)}

Training Set: {sum(train_counts)} epochs
Evaluation Set: {sum(eval_counts)} epochs
Split Ratio: {sum(train_counts)/(sum(train_counts)+sum(eval_counts)):.2f}

Most Common Class: {class_labels[np.argmax(counts)]} ({max(counts)} epochs)
Least Common Class: {class_labels[np.argmin(counts)]} ({min(counts)} epochs)
Class Imbalance Ratio: {max(counts)/min(counts):.2f}"""
    
    ax4.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
            fontdict={'family': 'monospace'})
    
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_training_progress(history, model_name):
    """Create detailed training progress visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['eval_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training Progress (Loss)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history['eval_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Training Progress (Accuracy)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Learning curves with smoothing
    if len(epochs) > 5:
        from scipy.ndimage import uniform_filter1d
        smoothed_train_loss = uniform_filter1d(history['train_loss'], size=min(5, len(epochs)//2))
        smoothed_eval_loss = uniform_filter1d(history['eval_loss'], size=min(5, len(epochs)//2))
        
        ax3.plot(epochs, history['train_loss'], 'b-', alpha=0.3, label='Raw Training Loss')
        ax3.plot(epochs, smoothed_train_loss, 'b-', linewidth=2, label='Smoothed Training Loss')
        ax3.plot(epochs, history['eval_loss'], 'r-', alpha=0.3, label='Raw Validation Loss')
        ax3.plot(epochs, smoothed_eval_loss, 'r-', linewidth=2, label='Smoothed Validation Loss')
    else:
        ax3.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        ax3.plot(epochs, history['eval_loss'], 'r-', linewidth=2, label='Validation Loss')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'{model_name} - Smoothed Learning Curves', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training statistics
    ax4.axis('off')
    
    final_train_acc = history['train_acc'][-1]
    final_eval_acc = history['eval_acc'][-1]
    best_eval_acc = max(history['eval_acc'])
    best_epoch = history['eval_acc'].index(best_eval_acc) + 1
    final_train_loss = history['train_loss'][-1]
    final_eval_loss = history['eval_loss'][-1]
    min_eval_loss = min(history['eval_loss'])
    
    # Calculate overfitting measure
    overfitting = final_train_acc - final_eval_acc
    
    stats_text = f"""Training Statistics

Total Epochs: {len(epochs)}
Best Validation Accuracy: {best_eval_acc:.2f}% (Epoch {best_epoch})
Final Training Accuracy: {final_train_acc:.2f}%
Final Validation Accuracy: {final_eval_acc:.2f}%

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_eval_loss:.4f}
Minimum Validation Loss: {min_eval_loss:.4f}

Overfitting Measure: {overfitting:.2f}%
{'⚠️ Potential Overfitting' if overfitting > 10 else '✅ Good Generalization'}

Accuracy Improvement: {best_eval_acc - history['eval_acc'][0]:.2f}%
Loss Reduction: {(history['eval_loss'][0] - min_eval_loss)/history['eval_loss'][0]*100:.1f}%"""
    
    ax4.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
            fontdict={'family': 'monospace'})
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_progress.png', 
               dpi=300, bbox_inches='tight')
    plt.show()


def visualize_results(history, results, model_name, class_names):
    """Create comprehensive visualization of results."""
    # Get actual classes present in data
    actual_classes = sorted(list(set(results['labels']) | set(results['predictions'])))
    actual_class_names = [class_names[i] for i in actual_classes]
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Accuracy curves  
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['eval_acc'], 'r-', label='Eval Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(results['labels'], results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=actual_class_names, yticklabels=actual_class_names, ax=ax3)
    ax3.set_title(f'{model_name} - Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # 4. Normalized Confusion Matrix
    ax4 = fig.add_subplot(gs[0, 3])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=actual_class_names, yticklabels=actual_class_names, ax=ax4)
    ax4.set_title(f'{model_name} - Normalized CM', fontweight='bold')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    # 5. Per-class metrics
    ax5 = fig.add_subplot(gs[1, 0])
    precision, recall, f1, support = precision_recall_fscore_support(
        results['labels'], results['predictions'], average=None, zero_division=0
    )
    
    x = np.arange(len(actual_classes))
    width = 0.25
    
    ax5.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    ax5.bar(x, recall, width, label='Recall', alpha=0.8, color='lightgreen')
    ax5.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='salmon')
    
    ax5.set_xlabel('Sleep Stage')
    ax5.set_ylabel('Score')
    ax5.set_title(f'{model_name} - Per-Class Metrics', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(actual_class_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1.1)
    
    # 6. Class distribution comparison
    ax6 = fig.add_subplot(gs[1, 1])
    unique_true, counts_true = np.unique(results['labels'], return_counts=True)
    unique_pred, counts_pred = np.unique(results['predictions'], return_counts=True)
    
    # Create arrays for all possible classes
    all_true_counts = np.zeros(len(actual_classes))
    all_pred_counts = np.zeros(len(actual_classes))
    
    for i, class_id in enumerate(actual_classes):
        if class_id in unique_true:
            all_true_counts[i] = counts_true[unique_true == class_id][0]
        if class_id in unique_pred:
            all_pred_counts[i] = counts_pred[unique_pred == class_id][0]
    
    x = np.arange(len(actual_classes))
    width = 0.35
    
    ax6.bar(x - width/2, all_true_counts, width, label='True', alpha=0.8, color='steelblue')
    ax6.bar(x + width/2, all_pred_counts, width, label='Predicted', alpha=0.8, color='darkorange')
    
    ax6.set_xlabel('Sleep Stage')
    ax6.set_ylabel('Count')
    ax6.set_title(f'{model_name} - Class Distribution', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(actual_class_names, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Confidence distribution
    ax7 = fig.add_subplot(gs[1, 2])
    max_probs = np.max(results['probabilities'], axis=1)
    ax7.hist(max_probs, bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax7.set_xlabel('Prediction Confidence')
    ax7.set_ylabel('Count')
    ax7.set_title(f'{model_name} - Confidence Distribution', fontweight='bold')
    ax7.axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(max_probs):.3f}')
    ax7.axvline(np.median(max_probs), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(max_probs):.3f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Error analysis
    ax8 = fig.add_subplot(gs[1, 3])
    errors = results['predictions'] != results['labels']
    if np.any(errors):
        error_indices = np.where(errors)[0]
        error_true = results['labels'][error_indices]
        error_pred = results['predictions'][error_indices]
        
        error_matrix = np.zeros((len(actual_classes), len(actual_classes)))
        for t, p in zip(error_true, error_pred):
            t_idx = actual_classes.index(t)
            p_idx = actual_classes.index(p)
            error_matrix[t_idx, p_idx] += 1
        
        if error_matrix.sum() > 0:
            sns.heatmap(error_matrix, annot=True, fmt='.0f', cmap='Reds',
                       xticklabels=actual_class_names, yticklabels=actual_class_names, ax=ax8)
            ax8.set_title(f'{model_name} - Error Matrix', fontweight='bold')
            ax8.set_xlabel('Predicted (Errors Only)')
            ax8.set_ylabel('True (Errors Only)')
        else:
            ax8.text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=20, fontweight='bold')
            ax8.set_title(f'{model_name} - Perfect Classification!', fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=20, fontweight='bold')
        ax8.set_title(f'{model_name} - Perfect Classification!', fontweight='bold')
    
    # 9. Summary statistics (spanning 2 columns)
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis('off')
    
    # Calculate additional metrics
    balanced_acc = np.mean([recall[i] for i in range(len(actual_classes))])
    macro_f1 = np.mean(f1)
    
    summary_text = f"""{model_name} - Comprehensive Performance Summary

═══ OVERALL METRICS ═══
Overall Accuracy: {results['accuracy']:.2f}%
Weighted F1 Score: {results['f1_score']:.4f}
Macro F1 Score: {macro_f1:.4f}
Balanced Accuracy: {balanced_acc:.4f}

═══ DATASET INFO ═══
Total Test Samples: {len(results['labels'])}
Correct Predictions: {np.sum(results['predictions'] == results['labels'])}
Incorrect Predictions: {np.sum(results['predictions'] != results['labels'])}
Classes Present: {len(actual_classes)} out of 5 possible

═══ CONFIDENCE METRICS ═══
Mean Confidence: {np.mean(max_probs):.3f}
Confidence Std: {np.std(max_probs):.3f}
Min Confidence: {np.min(max_probs):.3f}
Max Confidence: {np.max(max_probs):.3f}

═══ PER-CLASS F1 SCORES ═══"""
    
    for i, class_id in enumerate(actual_classes):
        summary_text += f"\n{actual_class_names[i]} (Class {class_id}): {f1[i]:.3f}"
        
    ax9.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
            fontdict={'family': 'monospace'}, transform=ax9.transAxes)
    
    # 10. Training info (spanning 2 columns)
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    training_info = f"""{model_name} - Training Information

═══ TRAINING PROGRESS ═══
Total Epochs Trained: {len(history['train_acc'])}
Best Validation Accuracy: {max(history['eval_acc']):.2f}%
Best Epoch: {history['eval_acc'].index(max(history['eval_acc'])) + 1}
Final Training Accuracy: {history['train_acc'][-1]:.2f}%
Final Validation Accuracy: {history['eval_acc'][-1]:.2f}%

═══ LOSS INFORMATION ═══
Final Training Loss: {history['train_loss'][-1]:.4f}
Final Validation Loss: {history['eval_loss'][-1]:.4f}
Minimum Validation Loss: {min(history['eval_loss']):.4f}
Best Loss Epoch: {history['eval_loss'].index(min(history['eval_loss'])) + 1}

═══ CONVERGENCE ANALYSIS ═══
Training-Validation Gap: {history['train_acc'][-1] - history['eval_acc'][-1]:.2f}%
Overfitting Status: {'⚠️ Possible Overfitting' if (history['train_acc'][-1] - history['eval_acc'][-1]) > 10 else '✅ Good Generalization'}
Accuracy Improvement: {max(history['eval_acc']) - history['eval_acc'][0]:.2f}%
Loss Reduction: {(history['eval_loss'][0] - min(history['eval_loss']))/history['eval_loss'][0]*100:.1f}%"""
    
    ax10.text(0.05, 0.95, training_info, fontsize=11, verticalalignment='top',
             fontdict={'family': 'monospace'}, transform=ax10.transAxes)
    
    plt.suptitle(f'{model_name} - Complete Performance Analysis', fontsize=16, fontweight='bold')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_complete_results.png', 
               dpi=300, bbox_inches='tight')
    plt.show()


def print_detailed_report(results, model_name, class_names):
    """Print detailed classification report."""
    print("\n" + "="*80)
    print(f"DETAILED CLASSIFICATION REPORT - {model_name}")
    print("="*80)
    
    # Get actual classes present in the data
    actual_classes = sorted(list(set(results['labels']) | set(results['predictions'])))
    actual_class_names = [class_names[i] for i in actual_classes]
    
    print(f"Classes present in data: {actual_class_names}")
    print(f"Total classes in dataset: {len(actual_classes)} out of {len(class_names)} possible")
    print()
    
    try:
        print(classification_report(results['labels'], results['predictions'], 
                                  target_names=actual_class_names, digits=4, zero_division=0))
    except Exception as e:
        print(f"Error generating classification report: {e}")
        # Fallback: basic accuracy
        accuracy = np.mean(results['predictions'] == results['labels'])
        print(f"Basic accuracy: {accuracy:.4f}")
    
    # Additional detailed analysis
    print(f"\nDetailed Analysis:")
    print("-" * 50)
    unique_labels, label_counts = np.unique(results['labels'], return_counts=True)
    unique_preds, pred_counts = np.unique(results['predictions'], return_counts=True)
    
    print(f"Ground Truth Distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  {class_names[label]} (Class {label}): {count} samples ({count/len(results['labels'])*100:.1f}%)")
    
    print(f"\nPrediction Distribution:")
    for pred, count in zip(unique_preds, pred_counts):
        print(f"  {class_names[pred]} (Class {pred}): {count} samples ({count/len(results['predictions'])*100:.1f}%)")
    
    # Calculate per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for class_id in actual_classes:
        class_mask = results['labels'] == class_id
        if np.any(class_mask):
            class_acc = np.mean(results['predictions'][class_mask] == results['labels'][class_mask])
            print(f"  {class_names[class_id]} (Class {class_id}): {class_acc:.4f} ({class_acc*100:.2f}%)")


def create_comparison_plot(results_summary):
    """Create comparison plot between models."""
    if len(results_summary) < 2:
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    model_names = []
    accuracies = []
    f1_scores = []
    params = []
    
    for model_name, metrics in results_summary.items():
        if 'error' not in metrics:
            model_names.append(model_name)
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            params.append(metrics['trainable_params'])
    
    if not model_names:
        return
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # F1 Score comparison
    bars2 = ax2.bar(model_names, f1_scores, color=colors)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Model F1 Score Comparison', fontweight='bold')
    ax2.set_ylim(0, 1)
    
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Parameter count comparison
    bars3 = ax3.bar(model_names, [p/1000 for p in params], color=colors)
    ax3.set_ylabel('Parameters (K)')
    ax3.set_title('Model Complexity Comparison', fontweight='bold')
    
    for bar, param in zip(bars3, params):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(params)/1000*0.02,
                f'{param/1000:.1f}K', ha='center', va='bottom', fontweight='bold')
    
    # Summary comparison
    ax4.axis('off')
    
    # Find best model
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
    most_efficient_idx = np.argmin(params)
    
    comparison_text = f"""Model Comparison Summary

🏆 BEST ACCURACY: {model_names[best_acc_idx]}
   Accuracy: {accuracies[best_acc_idx]:.2f}%
   
🎯 BEST F1 SCORE: {model_names[best_f1_idx]}
   F1 Score: {f1_scores[best_f1_idx]:.4f}
   
⚡ MOST EFFICIENT: {model_names[most_efficient_idx]}
   Parameters: {params[most_efficient_idx]:,}
   
📊 OVERALL RANKING:"""
    
    # Calculate overall score (accuracy + f1)
    overall_scores = [(acc + f1*100)/2 for acc, f1 in zip(accuracies, f1_scores)]
    ranking = sorted(zip(model_names, overall_scores), key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(ranking):
        comparison_text += f"\n   {i+1}. {name}: {score:.2f}"
    
    ax4.text(0.1, 0.9, comparison_text, fontsize=12, verticalalignment='top',
            fontdict={'family': 'monospace'})
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ================================
# Main Test Function
# ================================

def main():
    """Main function to test Option C models with comprehensive analysis."""
    
    # Configuration
    edf_file = r"C:\Users\21358\Desktop\01_sleep_psg.edf"
    mat_file = r"C:\Users\21358\Desktop\01_SleepStages.mat"
    
    # Check if files exist
    if not os.path.exists(edf_file):
        logger.error(f"EDF file not found: {edf_file}")
        return
    if not os.path.exists(mat_file):
        logger.error(f"MAT file not found: {mat_file}")
        return
    
    # Model configurations
    config = {
        'n_reservoir': 512,
        'tau': 0.02,
        'threshold': 1.0,
        'batch_size': 32,
        'n_epochs': 25,  # Reasonable for testing
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    }
    
    stage_names = ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
    
    try:
        # Load and preprocess data
        dataset, n_channels, n_classes, sfreq, global_mean, global_std = load_and_preprocess_data(
            edf_file, mat_file, use_mne_cleaning=True
        )
        
        # Create stratified split
        logger.info("Creating stratified train/eval split...")
        train_indices, eval_indices = create_stratified_split(
            dataset, test_size=0.2, random_state=42
        )
        
        # Debug class distribution
        class_debug_info = debug_data_classes(dataset, train_indices, eval_indices)
        
        # Adjust n_classes based on actual data
        actual_n_classes = class_debug_info['n_classes_total']
        if actual_n_classes != n_classes:
            logger.warning(f"Adjusting n_classes from {n_classes} to {actual_n_classes} based on actual data")
            n_classes = actual_n_classes
        
        logger.info(f"Dataset split:")
        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Evaluation samples: {len(eval_indices)}")
        logger.info(f"Actual number of classes: {n_classes}")
        
        # Create data overview visualization
        create_data_overview_plot(class_debug_info, stage_names)
        
        # Initialize pipeline
        pipeline = OptionCPipeline()
        
        # Create data loaders
        train_loader = pipeline.create_dataloader(
            dataset, train_indices, global_mean, global_std,
            config['batch_size'], shuffle=True
        )
        eval_loader = pipeline.create_dataloader(
            dataset, eval_indices, global_mean, global_std,
            config['batch_size'], shuffle=False
        )
        
        # Results storage
        results_summary = {}
        
        # Test Original Model
        logger.info("\n" + "="*70)
        logger.info("TESTING OPTION C: Original CNN-Reservoir Model")
        logger.info("="*70)
        
        try:
            model_original = SimplifiedSNNModel(
                n_channels, config['n_reservoir'], n_classes, sfreq,
                config['tau'], config['threshold']
            ).to(pipeline.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model_original.parameters())
            trainable_params = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
            logger.info(f"Original Model - Total parameters: {total_params:,}")
            logger.info(f"Original Model - Trainable parameters: {trainable_params:,}")
            
            # Train model
            model_original, history_original, best_acc_original = pipeline.train_model(
                model_original, train_loader, eval_loader,
                config['n_epochs'], config['learning_rate'], config['weight_decay'],
                "Original CNN-Reservoir", use_enhanced=False
            )
            
            # Load best model and evaluate
            model_filename = 'best_original_cnn_reservoir.pth'
            model_original.load_state_dict(torch.load(model_filename, weights_only=True))
            results_original = pipeline.evaluate_final(model_original, eval_loader, use_enhanced=False)
            
            # Store results
            results_summary['Original CNN-Reservoir'] = {
                'accuracy': results_original['accuracy'],
                'f1_score': results_original['f1_score'],
                'best_acc': best_acc_original,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            # Visualize training progress
            visualize_training_progress(history_original, "Original CNN-Reservoir")
            
            # Visualize detailed results
            visualize_results(history_original, results_original, "Original CNN-Reservoir", stage_names)
            
            # Print detailed report
            print_detailed_report(results_original, "Original CNN-Reservoir", stage_names)
            
        except Exception as e:
            logger.error(f"Original model failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results_summary['Original CNN-Reservoir'] = {'error': str(e)}
        
        # Test Enhanced Model
        logger.info("\n" + "="*70)
        logger.info("TESTING OPTION C: Enhanced CNN-Reservoir with Attention")
        logger.info("="*70)
        
        try:
            model_enhanced = EnhancedSNNModel(
                n_channels, config['n_reservoir'], n_classes, sfreq,
                config['tau'], config['threshold']
            ).to(pipeline.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model_enhanced.parameters())
            trainable_params = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
            logger.info(f"Enhanced Model - Total parameters: {total_params:,}")
            logger.info(f"Enhanced Model - Trainable parameters: {trainable_params:,}")
            
            # Train model
            model_enhanced, history_enhanced, best_acc_enhanced = pipeline.train_model(
                model_enhanced, train_loader, eval_loader,
                config['n_epochs'], config['learning_rate'], config['weight_decay'],
                "Enhanced CNN-Attention", use_enhanced=True
            )
            
            # Load best model and evaluate
            model_filename = 'best_enhanced_cnn_attention.pth'
            model_enhanced.load_state_dict(torch.load(model_filename, weights_only=True))
            results_enhanced = pipeline.evaluate_final(model_enhanced, eval_loader, use_enhanced=True)
            
            # Store results
            results_summary['Enhanced CNN-Attention'] = {
                'accuracy': results_enhanced['accuracy'],
                'f1_score': results_enhanced['f1_score'],
                'best_acc': best_acc_enhanced,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            # Visualize training progress
            visualize_training_progress(history_enhanced, "Enhanced CNN-Attention")
            
            # Visualize detailed results
            visualize_results(history_enhanced, results_enhanced, "Enhanced CNN-Attention", stage_names)
            
            # Print detailed report
            print_detailed_report(results_enhanced, "Enhanced CNN-Attention", stage_names)
            
        except Exception as e:
            logger.error(f"Enhanced model failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results_summary['Enhanced CNN-Attention'] = {'error': str(e)}
        
        # Create comparison visualization
        create_comparison_plot(results_summary)
        
        # Final comparison and analysis
        logger.info("\n" + "="*70)
        logger.info("OPTION C FINAL COMPARISON AND ANALYSIS")
        logger.info("="*70)
        
        print(f"\n{'='*85}")
        print(f"{'COMPREHENSIVE OPTION C TEST RESULTS':^85}")
        print(f"{'='*85}")
        
        print(f"\n{'Model':<25} {'Accuracy (%)':<15} {'F1 Score':<15} {'Parameters':<15} {'Status':<15}")
        print("-" * 85)
        
        successful_models = []
        for model_name, metrics in results_summary.items():
            if 'error' in metrics:
                print(f"{model_name:<25} {'-':<15} {'-':<15} {'-':<15} {'❌ Failed':<15}")
                print(f"{'Error:':<25} {metrics['error']}")
            else:
                print(f"{model_name:<25} {metrics['accuracy']:<15.2f} {metrics['f1_score']:<15.4f} "
                      f"{metrics['trainable_params']:<15,} {'✅ Success':<15}")
                successful_models.append((model_name, metrics))
        
        # Detailed analysis if we have successful models
        if successful_models:
            print(f"\n{'DETAILED PERFORMANCE ANALYSIS':^85}")
            print("=" * 85)
            
            best_accuracy = max(successful_models, key=lambda x: x[1]['accuracy'])
            best_f1 = max(successful_models, key=lambda x: x[1]['f1_score'])
            most_efficient = min(successful_models, key=lambda x: x[1]['trainable_params'])
            
            print(f"\n🏆 BEST ACCURACY: {best_accuracy[0]}")
            print(f"   └─ Accuracy: {best_accuracy[1]['accuracy']:.2f}%")
            print(f"   └─ F1 Score: {best_accuracy[1]['f1_score']:.4f}")
            print(f"   └─ Parameters: {best_accuracy[1]['trainable_params']:,}")
            
            print(f"\n🎯 BEST F1 SCORE: {best_f1[0]}")
            print(f"   └─ F1 Score: {best_f1[1]['f1_score']:.4f}")
            print(f"   └─ Accuracy: {best_f1[1]['accuracy']:.2f}%")
            print(f"   └─ Parameters: {best_f1[1]['trainable_params']:,}")
            
            print(f"\n⚡ MOST EFFICIENT: {most_efficient[0]}")
            print(f"   └─ Parameters: {most_efficient[1]['trainable_params']:,}")
            print(f"   └─ Accuracy: {most_efficient[1]['accuracy']:.2f}%")
            print(f"   └─ F1 Score: {most_efficient[1]['f1_score']:.4f}")
            
            # Performance insights
            if len(successful_models) >= 2:
                print(f"\n📊 COMPARATIVE INSIGHTS:")
                acc_diff = abs(successful_models[0][1]['accuracy'] - successful_models[1][1]['accuracy'])
                f1_diff = abs(successful_models[0][1]['f1_score'] - successful_models[1][1]['f1_score'])
                param_diff = abs(successful_models[0][1]['trainable_params'] - successful_models[1][1]['trainable_params'])
                
                print(f"   └─ Accuracy difference: {acc_diff:.2f}%")
                print(f"   └─ F1 Score difference: {f1_diff:.4f}")
                print(f"   └─ Parameter difference: {param_diff:,}")
                
                if acc_diff < 2.0:
                    print(f"   └─ 📋 Similar performance between models")
                elif successful_models[0][1]['accuracy'] > successful_models[1][1]['accuracy']:
                    print(f"   └─ 🥇 {successful_models[0][0]} performs better")
                else:
                    print(f"   └─ 🥇 {successful_models[1][0]} performs better")
        
        # Save comprehensive summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON
        save_data = {
            'timestamp': timestamp,
            'config': config,
            'dataset_info': {
                'n_channels': n_channels,
                'n_classes': n_classes,
                'sfreq': sfreq,
                'total_epochs': len(dataset),
                'train_epochs': len(train_indices),
                'eval_epochs': len(eval_indices)
            },
            'class_debug_info': class_debug_info,
            'results_summary': results_summary
        }
        
        with open(f'option_c_complete_results_{timestamp}.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def serialize_dict(d):
                if isinstance(d, dict):
                    return {k: serialize_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [serialize_dict(item) for item in d]
                else:
                    return convert_numpy(d)
            
            json.dump(serialize_dict(save_data), f, indent=2)
        
        # Save text summary
        with open(f'option_c_summary_{timestamp}.txt', 'w') as f:
            f.write("OPTION C COMPREHENSIVE TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(edf_file)}\n")
            f.write(f"Channels: {n_channels}\n")
            f.write(f"Classes: {n_classes}\n")
            f.write(f"Sampling Rate: {sfreq} Hz\n")
            f.write(f"Training Epochs: {len(train_indices)}\n")
            f.write(f"Evaluation Epochs: {len(eval_indices)}\n\n")
            
            f.write("RESULTS BY MODEL:\n")
            f.write("-" * 30 + "\n")
            for model_name, metrics in results_summary.items():
                f.write(f"\n{model_name}:\n")
                if 'error' in metrics:
                    f.write(f"  Status: FAILED\n")
                    f.write(f"  Error: {metrics['error']}\n")
                else:
                    f.write(f"  Status: SUCCESS\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.2f}%\n")
                    f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
                    f.write(f"  Best Training Accuracy: {metrics['best_acc']:.2f}%\n")
                    f.write(f"  Total Parameters: {metrics['total_params']:,}\n")
                    f.write(f"  Trainable Parameters: {metrics['trainable_params']:,}\n")
        
        print(f"\n{'='*85}")
        print(f"✅ OPTION C TESTING COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved with timestamp: {timestamp}")
        print(f"📊 Generated visualizations:")
        print(f"   └─ data_overview.png")
        print(f"   └─ *_training_progress.png")
        print(f"   └─ *_complete_results.png")
        print(f"   └─ model_comparison.png")
        print(f"📋 Detailed reports:")
        print(f"   └─ option_c_complete_results_{timestamp}.json")
        print(f"   └─ option_c_summary_{timestamp}.txt")
        print(f"{'='*85}")
        
        # Final recommendations
        if successful_models:
            print(f"\n🔍 RECOMMENDATIONS:")
            if len(successful_models) == 1:
                print(f"   └─ Only one model succeeded: {successful_models[0][0]}")
                print(f"   └─ Consider investigating why the other model failed")
            else:
                best_overall = max(successful_models, key=lambda x: (x[1]['accuracy'] + x[1]['f1_score']*100)/2)
                print(f"   └─ Recommended model: {best_overall[0]}")
                print(f"   └─ Rationale: Best overall performance")
                
                if best_overall[1]['trainable_params'] > 1000000:
                    print(f"   └─ ⚠️  Consider model complexity vs. performance trade-off")
                if best_overall[1]['accuracy'] > 90:
                    print(f"   └─ 🎉 Excellent performance achieved!")
                elif best_overall[1]['accuracy'] > 80:
                    print(f"   └─ 👍 Good performance, consider hyperparameter tuning")
                else:
                    print(f"   └─ 🔧 Performance could be improved - check data quality and model architecture")
        
        logger.info("Option C comprehensive testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Critical error during Option C testing: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'option_c_error_log_{timestamp}.txt', 'w') as f:
            f.write(f"Option C Test Error Log\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Full Traceback:\n")
            f.write(traceback.format_exc())
        
        print(f"\n❌ CRITICAL ERROR OCCURRED")
        print(f"📋 Error log saved: option_c_error_log_{timestamp}.txt")
        print(f"🔧 Please check the error log for debugging information")


if __name__ == "__main__":
    print("="*85)
    print("🧠 OPTION C COMPREHENSIVE TEST SUITE")
    print("🚀 Testing CNN-Reservoir Models for Sleep Staging")
    print("="*85)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*85)
    
    main()
    
    print("\n" + "="*85)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Thank you for using the Option C Test Suite!")
    print("="*85)