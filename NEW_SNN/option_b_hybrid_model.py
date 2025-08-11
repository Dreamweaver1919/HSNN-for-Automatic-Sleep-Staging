"""
Option B: Hybrid Model (CNN + Population Encoding + Reservoir)
==============================================================
Combines CNN features with population-encoded handcrafted features.
Best of both worlds: automatic feature learning + domain knowledge.
"""
import warnings
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from spikingjelly.activation_based import neuron, surrogate
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# Feature Extraction (Same as Option A)
# ================================

class FastFeatureExtractor:
    """Optimized feature extractor - much faster than original."""
    
    def __init__(self, sfreq: float = 200.0, use_simple_features: bool = True):
        self.sfreq = sfreq
        self.use_simple_features = use_simple_features
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }
        
    def extract_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """Extract features FAST - skip expensive computations."""
        
        if self.use_simple_features:
            return self.extract_simple_features(eeg_window)
        else:
            return self.extract_full_features(eeg_window)
    
    def extract_simple_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """Extract only simple, fast features."""
        all_features = []
        
        for ch in range(eeg_window.shape[0]):
            channel_data = eeg_window[ch]
            channel_features = []
            
            # Basic statistics (5 features) - FAST
            channel_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.ptp(channel_data)  # peak-to-peak
            ])
            
            # Simple percentiles (2 features) - FAST
            channel_features.extend([
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
            
            # Basic frequency domain (5 features) - Simplified
            # Use FFT instead of Welch for speed
            fft_vals = np.abs(np.fft.rfft(channel_data))
            freqs = np.fft.rfftfreq(len(channel_data), 1/self.sfreq)
            
            total_power = np.sum(fft_vals)
            if total_power > 0:
                for band_name, (low, high) in self.bands.items():
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    band_power = np.sum(fft_vals[idx]) / total_power
                    channel_features.append(band_power)
            else:
                channel_features.extend([0.2] * 5)  # Equal distribution
            
            # Zero-crossing rate (1 feature) - FAST
            signs = np.sign(channel_data - np.mean(channel_data))
            zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(channel_data))
            channel_features.append(zcr)
            
            all_features.extend(channel_features)
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_full_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """Extract full features (slower but more comprehensive)."""
        all_features = []
        
        for ch in range(eeg_window.shape[0]):
            channel_data = eeg_window[ch]
            channel_features = []
            
            # Time domain features (7) - with safety checks
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                
                # Safe skewness and kurtosis
                if std_val > 1e-10:
                    try:
                        skew_val = skew(channel_data)
                        kurt_val = kurtosis(channel_data)
                        if np.isnan(skew_val) or np.isinf(skew_val):
                            skew_val = 0.0
                        if np.isnan(kurt_val) or np.isinf(kurt_val):
                            kurt_val = 0.0
                    except:
                        skew_val = kurt_val = 0.0
                else:
                    skew_val = kurt_val = 0.0
                
                channel_features.extend([
                    mean_val,
                    std_val,
                    skew_val,
                    kurt_val,
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.ptp(channel_data)
                ])
            
            # Hjorth parameters (3) - FAST version
            diff1 = np.diff(channel_data)
            activity = np.var(channel_data)
            
            if activity > 1e-10:
                mobility = np.sqrt(np.var(diff1) / activity)
                diff2 = np.diff(diff1)
                if np.var(diff1) > 1e-10:
                    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / (mobility + 1e-10)
                else:
                    complexity = 0.0
            else:
                mobility = complexity = 0.0
            
            channel_features.extend([activity, mobility, complexity])
            
            # Frequency features (6) - using FFT for speed
            fft_vals = np.abs(np.fft.rfft(channel_data))
            freqs = np.fft.rfftfreq(len(channel_data), 1/self.sfreq)
            
            total_power = np.sum(fft_vals)
            if total_power > 0:
                for band_name, (low, high) in self.bands.items():
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    band_power = np.sum(fft_vals[idx]) / total_power
                    channel_features.append(band_power)
            else:
                channel_features.extend([0.2] * 5)
            
            # Spectral edge frequency
            if total_power > 0:
                cumsum_power = np.cumsum(fft_vals)
                idx_95 = np.where(cumsum_power >= 0.95 * total_power)[0]
                sef_95 = freqs[idx_95[0]] if len(idx_95) > 0 else freqs[-1]
            else:
                sef_95 = self.sfreq / 4
            channel_features.append(sef_95)
            
            # Skip sample entropy - it's VERY slow
            # Instead use simpler complexity measure
            channel_features.append(0.0)  # Placeholder for sample entropy
            
            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(channel_data - np.mean(channel_data))))) / (2 * len(channel_data))
            channel_features.append(zcr)
            
            all_features.extend(channel_features)
        
        return np.array(all_features, dtype=np.float32)


# ================================
# Population Encoding
# ================================

class PopulationEncoder:
    """Convert scalar features to population-coded spike trains."""
    
    def __init__(self, n_neurons_per_feature: int = 40, timesteps: int = 50, 
                 sigma: float = 0.15, sparse_factor: float = 0.8):
        self.n_neurons_per_feature = n_neurons_per_feature
        self.timesteps = timesteps
        self.sigma = sigma
        self.sparse_factor = sparse_factor
        self.feature_centers = np.linspace(0, 1, n_neurons_per_feature)
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode feature vector into spike trains."""
        n_features = len(features)
        total_neurons = n_features * self.n_neurons_per_feature
        spike_train = np.zeros((total_neurons, self.timesteps))
        
        # Normalize features
        features_norm = np.clip(features, -3, 3)
        features_norm = (features_norm - np.min(features_norm)) / (np.max(features_norm) - np.min(features_norm) + 1e-8)
        
        for i, value in enumerate(features_norm):
            start_idx = i * self.n_neurons_per_feature
            
            # Gaussian tuning curves
            activations = np.exp(-(value - self.feature_centers)**2 / (2 * self.sigma**2))
            activations = activations / (np.max(activations) + 1e-8)
            
            # Convert to spikes
            for j in range(self.n_neurons_per_feature):
                firing_rate = activations[j] * self.sparse_factor
                n_spikes = int(firing_rate * self.timesteps)
                
                if n_spikes > 0:
                    spike_times = np.random.choice(self.timesteps, 
                                                  size=min(n_spikes, self.timesteps),
                                                  replace=False)
                    spike_train[start_idx + j, spike_times] = 1
                    
        return spike_train


# ================================
# Reservoir Components
# ================================

class SpikingReservoir(nn.Module):
    """Fixed-weight spiking reservoir."""
    
    def __init__(self, n_input, n_reservoir, sfreq, tau=0.02, threshold=1.0):
        super().__init__()
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.sfreq = sfreq
        
        # Fixed weight connections
        self.fc_in = nn.Linear(n_input, n_reservoir, bias=False)
        self.fc_rec = nn.Linear(n_reservoir, n_reservoir, bias=False)
        
        # Freeze weights
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
            
        # LIF neurons
        tau_timesteps = tau * sfreq
        self.lif = neuron.LIFNode(tau=tau_timesteps, surrogate_function=surrogate.ATan(), 
                                 detach_reset=True)
        self.lif.v_threshold = threshold
        
    def forward(self, x):
        """Process input through reservoir."""
        batch_size, _, time_steps = x.shape
        device = x.device
        spike_counts = torch.zeros(batch_size, self.n_reservoir, device=device)
        
        self.lif.reset()
        h = torch.zeros(batch_size, self.n_reservoir, device=device)
        
        for t in range(time_steps):
            input_t = x[:, :, t]
            i_in = self.fc_in(input_t) + self.fc_rec(h)
            h = self.lif(i_in)
            spike_counts += h
            
        return spike_counts / time_steps  # Return rates


# ================================
# Hybrid Model
# ================================

class HybridCNNPopulationSNN(nn.Module):
    """Hybrid model combining CNN and population encoding."""
    
    def __init__(self, n_channels, n_handcrafted_features, n_neurons_per_feature,
                 n_reservoir, n_classes, sfreq, timesteps=50):
        super().__init__()
        
        # CNN branch for automatic feature extraction
        self.cnn = nn.Sequential(
            # First block
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=10, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second block
            nn.Conv1d(64, 128, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third block
            nn.Conv1d(128, 128, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(timesteps)  # Ensure temporal alignment
        )
        
        # Population encoding branch (parameters stored, encoding done in forward)
        self.n_handcrafted_features = n_handcrafted_features
        self.n_neurons_per_feature = n_neurons_per_feature
        self.timesteps = timesteps
        
        # Combined input size for reservoir
        population_neurons = n_handcrafted_features * n_neurons_per_feature
        cnn_features = 128
        total_input = population_neurons + cnn_features
        
        # Spiking reservoir
        self.reservoir = SpikingReservoir(total_input, n_reservoir, sfreq)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_reservoir, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x, population_spikes):
        """
        x: raw EEG (batch_size, n_channels, time_steps)
        population_spikes: pre-encoded spikes (batch_size, n_pop_neurons, timesteps)
        """
        # CNN branch
        cnn_features = self.cnn(x)  # (batch_size, 128, timesteps)
        
        # Combine features
        combined = torch.cat([population_spikes, cnn_features], dim=1)
        
        # Process through reservoir
        reservoir_output = self.reservoir(combined)
        
        # Classify
        output = self.classifier(reservoir_output)
        
        return output


# ================================
# Dataset
# ================================

class HybridDataset(Dataset):
    """Dataset for hybrid model."""
    
    def __init__(self, raw_segments, features, labels, pop_encoder, global_mean, global_std):
        self.raw_segments = raw_segments
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pop_encoder = pop_encoder
        self.global_mean = global_mean
        self.global_std = global_std
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Raw EEG (normalized)
        raw = self.raw_segments[idx]
        raw = (raw - self.global_mean[:, None]) / self.global_std[:, None]
        raw_tensor = torch.tensor(raw, dtype=torch.float32)
        
        # Population-encoded features
        spike_train = self.pop_encoder.encode(self.features[idx])
        spike_tensor = torch.tensor(spike_train, dtype=torch.float32)
        
        return raw_tensor, spike_tensor, self.labels[idx]


# ================================
# Training Pipeline
# ================================

class HybridPipeline:
    """Complete pipeline for hybrid model."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def run(self, dataset, train_indices, eval_indices, global_mean, global_std, 
            n_channels, sfreq, config):
        """Run complete pipeline."""
        
        # 1. Extract features and prepare data
        logger.info("Step 1: Preparing data...")
        feature_extractor = FastFeatureExtractor(sfreq)
        
        # Extract for training
        train_raw, train_features, train_labels = [], [], []
        for idx in tqdm.tqdm(train_indices, desc="Processing train data"):
            segment, label = dataset[idx]
            train_raw.append(segment.numpy())
            features = feature_extractor.extract_features(segment.numpy())
            train_features.append(features)
            train_labels.append(label.item())
            
        # Extract for evaluation
        eval_raw, eval_features, eval_labels = [], [], []
        for idx in tqdm.tqdm(eval_indices, desc="Processing eval data"):
            segment, label = dataset[idx]
            eval_raw.append(segment.numpy())
            features = feature_extractor.extract_features(segment.numpy())
            eval_features.append(features)
            eval_labels.append(label.item())
            
        train_raw = np.array(train_raw)
        train_features = np.array(train_features)
        eval_raw = np.array(eval_raw)
        eval_features = np.array(eval_features)
        
        # 2. Normalize features
        logger.info("Step 2: Normalizing features...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        eval_features = scaler.transform(eval_features)
        
        n_features = train_features.shape[1]
        logger.info(f"Feature dimensions: {n_features}")
        
        # 3. Create population encoder
        logger.info("Step 3: Setting up population encoding...")
        pop_encoder = PopulationEncoder(
            n_neurons_per_feature=config['n_neurons_per_feature'],
            timesteps=config['timesteps'],
            sigma=config['sigma'],
            sparse_factor=config['sparse_factor']
        )
        
        # 4. Create datasets
        train_dataset = HybridDataset(train_raw, train_features, train_labels, 
                                    pop_encoder, global_mean, global_std)
        eval_dataset = HybridDataset(eval_raw, eval_features, eval_labels, 
                                   pop_encoder, global_mean, global_std)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=0)
        
        # 5. Create model
        logger.info("Step 4: Creating hybrid model...")
        model = HybridCNNPopulationSNN(
            n_channels=n_channels,
            n_handcrafted_features=n_features,
            n_neurons_per_feature=config['n_neurons_per_feature'],
            n_reservoir=config['n_reservoir'],
            n_classes=config['n_classes'],
            sfreq=sfreq,
            timesteps=config['timesteps']
        ).to(self.device)
        
        # 6. Training
        logger.info("Step 5: Training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['learning_rate'], 
            epochs=config['n_epochs'], steps_per_epoch=len(train_loader)
        )
        
        history = self.train(model, train_loader, eval_loader, criterion, 
                           optimizer, scheduler, config['n_epochs'])
        
        # 7. Final evaluation
        logger.info("Step 6: Final evaluation...")
        model.load_state_dict(torch.load('best_hybrid_model.pth'))
        results = self.evaluate_final(model, eval_loader)
        
        # 8. Visualize
        self.visualize_results(history, results, config['class_names'])
        
        return model, history, results
    
    def train(self, model, train_loader, eval_loader, criterion, optimizer, scheduler, n_epochs):
        """Training loop."""
        history = {
            'train_loss': [], 'train_acc': [],
            'eval_loss': [], 'eval_acc': []
        }
        best_acc = 0
        patience = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for raw, pop_spikes, labels in pbar:
                raw = raw.to(self.device)
                pop_spikes = pop_spikes.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(raw, pop_spikes)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'acc': f'{100.*train_correct/train_total:.2f}%'})
            
            # Evaluation
            eval_loss, eval_acc = self.evaluate(model, eval_loader, criterion)
            
            # Record history
            history['train_loss'].append(train_loss / train_total)
            history['train_acc'].append(100. * train_correct / train_total)
            history['eval_loss'].append(eval_loss)
            history['eval_acc'].append(eval_acc)
            
            print(f'Epoch {epoch+1}: Train Acc: {history["train_acc"][-1]:.2f}%, '
                  f'Eval Acc: {eval_acc:.2f}%')
            
            # Save best model
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(model.state_dict(), 'best_hybrid_model.pth')
                patience = 0
            else:
                patience += 1
                
            # Early stopping
            if patience >= 10:
                logger.info("Early stopping triggered!")
                break
                
        return history
    
    def evaluate(self, model, loader, criterion):
        """Evaluate model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for raw, pop_spikes, labels in loader:
                raw = raw.to(self.device)
                pop_spikes = pop_spikes.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(raw, pop_spikes)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / total, 100. * correct / total
    
    def evaluate_final(self, model, loader):
        """Final detailed evaluation."""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for raw, pop_spikes, labels in loader:
                raw = raw.to(self.device)
                pop_spikes = pop_spikes.to(self.device)
                
                outputs = model(raw, pop_spikes)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'accuracy': 100. * np.mean(np.array(all_preds) == np.array(all_labels)),
            'f1_score': f1_score(all_labels, all_preds, average='weighted')
        }
    
    def visualize_results(self, history, results, class_names):
        """Visualize training history and results."""
        fig = plt.figure(figsize=(15, 10))
        
        # Training curves
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['eval_loss'], label='Eval Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['eval_acc'], label='Eval Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax3 = plt.subplot(2, 2, 3)
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # Feature importance (placeholder)
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        summary_text = f"""Final Results:
        
Accuracy: {results['accuracy']:.2f}%
F1 Score: {results['f1_score']:.4f}

Model: Hybrid CNN + Population SNN
CNN Features: 128
Handcrafted Features: {18} per channel
Population Neurons: {40} per feature
Reservoir Size: {512}"""
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('option_b_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(results['labels'], results['predictions'], 
                                  target_names=class_names, digits=4))


# ================================
# Usage Example
# ================================

def use_option_b(dataset, train_indices, eval_indices, global_mean, global_std, 
                n_channels, sfreq):
    """Example usage of Option B."""
    
    config = {
        # Population encoding
        'n_neurons_per_feature': 40,
        'timesteps': 50,
        'sigma': 0.15,
        'sparse_factor': 0.8,
        
        # Model architecture
        'n_reservoir': 512,
        'n_classes': 5,
        
        # Training
        'batch_size': 16,  # Smaller due to dual inputs
        'n_epochs': 40,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        
        # Classes
        'class_names': ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
    }
    
    pipeline = HybridPipeline()
    model, history, results = pipeline.run(
        dataset, train_indices, eval_indices, global_mean, global_std,
        n_channels, sfreq, config
    )
    
    return model, results
