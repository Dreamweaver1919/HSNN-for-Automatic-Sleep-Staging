"""
Option C: Original Model (CNN → Reservoir)
==========================================
Your existing approach without population encoding.
Enhanced with better training strategies and visualization.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from spikingjelly.activation_based import neuron, surrogate
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# Model Components (from your notebook)
# ================================

class SpikingJellyReservoirSNN(nn.Module):
    """Fixed-weight reservoir using SpikingJelly LIF neurons."""
    
    def __init__(self, n_channels, n_reservoir, sfreq, tau=0.02, threshold=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_reservoir = n_reservoir
        self.sfreq = sfreq
        self.dt = 1.0 / sfreq
        self.tau = tau
        self.threshold = threshold

        # Fully-connected input and recurrent layers
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
            # Scale to control spectral radius
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
    """Complete SNN model with CNN feature extraction, reservoir, and classifier."""
    
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


# ================================
# Enhanced Model with Attention
# ================================

class EnhancedSNNModel(nn.Module):
    """Enhanced version with attention mechanism and residual connections."""
    
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
# Dataset
# ================================

class EEGDataset:
    """Dataset class compatible with the existing notebook structure."""
    
    def __init__(self, trials):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        segment, label = self.trials[idx]
        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return segment_tensor, label_tensor


# ================================
# Training Pipeline
# ================================

class OriginalModelPipeline:
    """Enhanced training pipeline for the original model."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def run(self, dataset, train_indices, eval_indices, global_mean, global_std,
            n_channels, n_classes, sfreq, config, use_enhanced=False):
        """Run the complete pipeline."""
        
        # 1. Create data loaders
        logger.info("Step 1: Creating data loaders...")
        train_loader = self.create_dataloader(
            dataset, train_indices, global_mean, global_std, 
            config['batch_size'], shuffle=True
        )
        eval_loader = self.create_dataloader(
            dataset, eval_indices, global_mean, global_std, 
            config['batch_size'], shuffle=False
        )
        
        # 2. Create model
        logger.info("Step 2: Creating model...")
        if use_enhanced:
            model = EnhancedSNNModel(
                n_channels, config['n_reservoir'], n_classes, sfreq,
                config['tau'], config['threshold']
            ).to(self.device)
            logger.info("Using enhanced model with attention")
        else:
            model = SimplifiedSNNModel(
                n_channels, config['n_reservoir'], n_classes, sfreq,
                config['tau'], config['threshold']
            ).to(self.device)
            logger.info("Using original model")
            
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # 3. Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
        
        # Learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 4. Training
        logger.info("Step 3: Training...")
        history = self.train(model, train_loader, eval_loader, criterion,
                           optimizer, scheduler, config['n_epochs'], use_enhanced)
        
        # 5. Final evaluation
        logger.info("Step 4: Final evaluation...")
        model.load_state_dict(torch.load('best_original_model.pth'))
        results = self.evaluate_final(model, eval_loader, use_enhanced)
        
        # 6. Visualize
        self.visualize_results(history, results, config['class_names'], use_enhanced)
        
        # 7. Analyze reservoir activity
        if not use_enhanced:
            self.analyze_reservoir(model, eval_loader, config['class_names'])
        
        return model, history, results
    
    def create_dataloader(self, dataset, indices, global_mean, global_std, 
                         batch_size, shuffle=True):
        """Create a DataLoader with normalization."""
        
        class NormalizedDataset(Dataset):
            def __init__(self, dataset, indices, global_mean, global_std):
                self.dataset = dataset
                self.indices = indices
                self.global_mean = global_mean.astype(np.float32)
                self.global_std = global_std.astype(np.float32)
                
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                segment, label = self.dataset[real_idx]
                # Ensure float32
                # if isinstance(segment, torch.Tensor):
                #     segment = segment.float()
                # else:
                #     segment = torch.tensor(segment, dtype=torch.float32)
                # Normalize
                segment = (segment - self.global_mean[:, None]) / self.global_std[:, None]
                return segment, label
        
        normalized_dataset = NormalizedDataset(dataset, indices, global_mean, global_std)
        return DataLoader(normalized_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, model, train_loader, eval_loader, criterion, optimizer, 
              scheduler, n_epochs, use_enhanced=False):
        """Enhanced training loop."""
        history = {
            'train_loss': [], 'train_acc': [],
            'eval_loss': [], 'eval_acc': []
        }
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
                
                # Update progress bar
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
            
            print(f'\nEpoch {epoch+1}: Train Loss: {history["train_loss"][-1]:.4f}, '
                  f'Train Acc: {history["train_acc"][-1]:.2f}%, '
                  f'Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%')
            
            # Save best model
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(model.state_dict(), 'best_original_model.pth')
                patience = 0
                print(f'✓ New best model saved! (Eval Acc: {eval_acc:.2f}%)')
            else:
                patience += 1
                
            # Early stopping
            if patience >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Check for convergence
            if history['train_acc'][-1] >= 85 and eval_acc >= 80:
                logger.info("Convergence criteria met!")
                break
                
        return history
    
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
            'f1_score': f1_score(all_labels, all_preds, average='weighted')
        }
    
    def visualize_results(self, history, results, class_names, use_enhanced):
        """Comprehensive visualization of results."""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training curves
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(history['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(history['eval_acc'], 'r-', label='Eval Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax3 = plt.subplot(3, 3, 3)
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # 3. Normalized Confusion Matrix
        ax4 = plt.subplot(3, 3, 4)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax4)
        ax4.set_title('Normalized Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        
        # 4. Per-class metrics
        ax5 = plt.subplot(3, 3, 5)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            results['labels'], results['predictions'], average=None
        )
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax5.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax5.bar(x, recall, width, label='Recall', alpha=0.8)
        ax5.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax5.set_xlabel('Sleep Stage')
        ax5.set_ylabel('Score')
        ax5.set_title('Per-Class Metrics')
        ax5.set_xticks(x)
        ax5.set_xticklabels(class_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 5. Class distribution in predictions
        ax6 = plt.subplot(3, 3, 6)
        unique_true, counts_true = np.unique(results['labels'], return_counts=True)
        unique_pred, counts_pred = np.unique(results['predictions'], return_counts=True)
        
        x = np.arange(len(class_names))
        width = 0.35
        
        ax6.bar(x - width/2, counts_true, width, label='True', alpha=0.8)
        ax6.bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.8)
        
        ax6.set_xlabel('Sleep Stage')
        ax6.set_ylabel('Count')
        ax6.set_title('Class Distribution')
        ax6.set_xticks(x)
        ax6.set_xticklabels(class_names, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 6. Confidence distribution
        ax7 = plt.subplot(3, 3, 7)
        max_probs = np.max(results['probabilities'], axis=1)
        ax7.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Prediction Confidence')
        ax7.set_ylabel('Count')
        ax7.set_title('Confidence Distribution')
        ax7.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 7. Error analysis
        ax8 = plt.subplot(3, 3, 8)
        errors = results['predictions'] != results['labels']
        error_indices = np.where(errors)[0]
        if len(error_indices) > 0:
            error_true = results['labels'][error_indices]
            error_pred = results['predictions'][error_indices]
            
            error_matrix = np.zeros((len(class_names), len(class_names)))
            for t, p in zip(error_true, error_pred):
                error_matrix[t, p] += 1
                
            sns.heatmap(error_matrix, annot=True, fmt='.0f', cmap='Reds',
                       xticklabels=class_names, yticklabels=class_names, ax=ax8)
            ax8.set_title('Error Matrix (True → Predicted)')
            ax8.set_xlabel('Predicted (Errors Only)')
            ax8.set_ylabel('True (Errors Only)')
        
        # 8. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        model_type = "Enhanced CNN-SNN with Attention" if use_enhanced else "Original CNN-Reservoir-SNN"
        summary_text = f"""Model Performance Summary
        
Model Type: {model_type}
Overall Accuracy: {results['accuracy']:.2f}%
Weighted F1 Score: {results['f1_score']:.4f}
Total Samples: {len(results['labels'])}
Correct Predictions: {np.sum(results['predictions'] == results['labels'])}
Errors: {np.sum(results['predictions'] != results['labels'])}

Per-Class F1 Scores:
"""
        for i, (cls, f1_val) in enumerate(zip(class_names, f1)):
            summary_text += f"\n{cls}: {f1_val:.3f}"
            
        ax9.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                fontdict={'family': 'monospace'})
        
        plt.tight_layout()
        filename = 'option_c_enhanced_results.png' if use_enhanced else 'option_c_original_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print("="*70)
        print(classification_report(results['labels'], results['predictions'], 
                                  target_names=class_names, digits=4))
    
    def analyze_reservoir(self, model, loader, class_names):
        """Analyze reservoir dynamics."""
        model.eval()
        
        # Collect reservoir activities for each class
        class_activities = {i: [] for i in range(len(class_names))}
        
        with torch.no_grad():
            for segments, labels in tqdm.tqdm(loader, desc="Analyzing reservoir"):
                segments = segments.to(self.device)
                
                # Get CNN features
                cnn_features = model.cnn(segments)
                
                # Get reservoir spike counts
                spike_counts = model.reservoir(cnn_features)
                
                # Group by class
                for spike_count, label in zip(spike_counts.cpu().numpy(), labels.numpy()):
                    class_activities[label].append(spike_count)
        
        # Visualize reservoir activity patterns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names):
            if i < len(axes):
                activities = np.array(class_activities[i])
                mean_activity = np.mean(activities, axis=0)
                std_activity = np.std(activities, axis=0)
                
                # Sort neurons by mean activity
                sorted_indices = np.argsort(mean_activity)[::-1][:50]  # Top 50 neurons
                
                ax = axes[i]
                x = np.arange(len(sorted_indices))
                ax.bar(x, mean_activity[sorted_indices], yerr=std_activity[sorted_indices],
                      capsize=2, alpha=0.7)
                ax.set_title(f'{class_name} - Top 50 Active Neurons')
                ax.set_xlabel('Neuron Index (sorted)')
                ax.set_ylabel('Mean Spike Count')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(class_names) < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle('Reservoir Activity Patterns by Sleep Stage', fontsize=16)
        plt.tight_layout()
        plt.savefig('reservoir_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Compute and visualize reservoir state similarity
        self.visualize_reservoir_similarity(class_activities, class_names)
    
    def visualize_reservoir_similarity(self, class_activities, class_names):
        """Visualize similarity between reservoir states for different classes."""
        # Compute mean activity for each class
        mean_activities = []
        for i in range(len(class_names)):
            activities = np.array(class_activities[i])
            mean_activities.append(np.mean(activities, axis=0))
        
        mean_activities = np.array(mean_activities)
        
        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(mean_activities)
        
        # Visualize
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=class_names, yticklabels=class_names,
                   vmin=0, vmax=1, center=0.5)
        plt.title('Reservoir State Similarity Between Sleep Stages')
        plt.tight_layout()
        plt.savefig('reservoir_similarity.png', dpi=300, bbox_inches='tight')
        plt.show()


# ================================
# Data Augmentation
# ================================

def augment_eeg(segment, noise_level=0.05, time_shift_range=50):
    """Simple EEG augmentation."""
    augmented = segment.clone()
    
    # Add noise
    if torch.rand(1) > 0.5:
        noise = torch.randn_like(segment) * noise_level * torch.std(segment)
        augmented += noise
    
    # Time shift
    if torch.rand(1) > 0.5:
        shift = torch.randint(-time_shift_range, time_shift_range, (1,)).item()
        augmented = torch.roll(augmented, shift, dims=-1)
    
    return augmented


# ================================
# Usage Examples
# ================================

def use_option_c(dataset, train_indices, eval_indices, global_mean, global_std,
                n_channels, n_classes, sfreq, use_enhanced=True):
    """Example usage of Option C."""
    
    config = {
        # Model parameters
        'n_reservoir': 512,
        'tau': 0.02,
        'threshold': 1.0,
        
        # Training parameters
        'batch_size': 32,
        'n_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        
        # Classes
        'class_names': ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
    }
    
    pipeline = OriginalModelPipeline()
    model, history, results = pipeline.run(
        dataset, train_indices, eval_indices, global_mean, global_std,
        n_channels, n_classes, sfreq, config, use_enhanced=use_enhanced
    )
    
    return model, results


def compare_all_options(dataset, train_indices, eval_indices, global_mean, global_std,
                       n_channels, n_classes, sfreq):
    """Compare all three options."""
    
    results_summary = {}
    
    # Option C - Original
    print("\n" + "="*70)
    print("Testing Option C: Original Model")
    print("="*70)
    model_c, results_c = use_option_c(
        dataset, train_indices, eval_indices, global_mean, global_std,
        n_channels, n_classes, sfreq, use_enhanced=False
    )
    results_summary['Original'] = {
        'accuracy': results_c['accuracy'],
        'f1_score': results_c['f1_score']
    }
    
    # Option C - Enhanced
    print("\n" + "="*70)
    print("Testing Option C: Enhanced Model with Attention")
    print("="*70)
    model_c_enhanced, results_c_enhanced = use_option_c(
        dataset, train_indices, eval_indices, global_mean, global_std,
        n_channels, n_classes, sfreq, use_enhanced=True
    )
    results_summary['Enhanced'] = {
        'accuracy': results_c_enhanced['accuracy'],
        'f1_score': results_c_enhanced['f1_score']
    }
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<15} {'F1 Score':<15}")
    print("-"*50)
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<20} {metrics['accuracy']:<15.2f} {metrics['f1_score']:<15.4f}")
    
    return results_summary