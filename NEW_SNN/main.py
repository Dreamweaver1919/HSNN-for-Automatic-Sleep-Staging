"""
Sleep Staging Model Comparison with MNE Bad Channel Detection
=============================================================
Complete pipeline with automatic bad channel detection using MNE.
"""

from scipy.io import loadmat
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, f1_score, precision_recall_fscore_support)
import pandas as pd
import tqdm
import logging
from tabulate import tabulate
import mne
import warnings

# Import the three option modules
from option_a_pure_population import PurePopulationPipeline, use_option_a
from option_b_hybrid_model import HybridPipeline, use_option_b
from option_c_original_model import OriginalModelPipeline, use_option_c

# Suppress warnings
warnings.filterwarnings('ignore', message='Precision loss occurred')
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EEGDataset(torch.utils.data.Dataset):
    """Dataset class for EEG data."""
    def __init__(self, trials):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        segment, label = self.trials[idx]
        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return segment_tensor, label_tensor


def load_and_preprocess_data(edf_file, mat_file, use_mne_cleaning=True):
    """
    Load EEG data and sleep stages with MNE-based bad channel detection.
    
    Args:
        edf_file: Path to EDF file
        mat_file: Path to MAT file with sleep stages
        use_mne_cleaning: Whether to use MNE's bad channel detection
        
    Returns:
        dataset, n_channels, n_classes, sfreq, global_mean, global_std
    """
    
    logger.info("="*70)
    logger.info("LOADING AND PREPROCESSING EEG DATA")
    logger.info("="*70)
    
    # Load EEG data using MNE
    logger.info(f"Loading EEG data from: {os.path.basename(edf_file)}")
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        # Remove non-EEG channels (like Event, ECG, EOG, etc.)
        logger.info(f"All channels in file: {raw.ch_names}")
        
        # Keep only EEG channels - remove common non-EEG channel names
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
        
        # Now work with only EEG channels
        logger.info(f"Sampling frequency: {sfreq} Hz")
        logger.info(f"EEG channels: {len(raw.ch_names)} channels")
        logger.info(f"Channel names: {raw.ch_names}")
        logger.info(f"Duration: {raw.times[-1]:.1f} seconds")
    except Exception as e:
        logger.error(f"Error reading EDF file: {e}")
        raise

    # Apply MNE's bad channel detection if requested
    if use_mne_cleaning and len(raw.ch_names) > 0:
        logger.info("\nApplying MNE bad channel detection...")
        
        # Find bad channels using MNE's methods
        original_bads = raw.info['bads'].copy() if 'bads' in raw.info else []
        
        # Method 1: Find flat channels
        try:
            flat_channels = mne.preprocessing.find_bad_channels_flat(raw, threshold=1e-15)[0]
            if flat_channels:
                # Verify channels exist before adding
                flat_channels = [ch for ch in flat_channels if ch in raw.ch_names]
                if flat_channels:
                    logger.info(f"Flat channels detected: {flat_channels}")
                    for ch in flat_channels:
                        if ch not in raw.info['bads']:
                            raw.info['bads'].append(ch)
        except Exception as e:
            logger.warning(f"Could not detect flat channels: {e}")
        
        # Method 2: Find noisy channels using correlation
        try:
            # Only use if we have enough channels
            if len(raw.ch_names) > 3:
                noisy_channels = mne.preprocessing.find_bad_channels_correlation(
                    raw, threshold=0.4, fraction_bad=0.3)[0]
                if noisy_channels:
                    # Verify channels exist before adding
                    noisy_channels = [ch for ch in noisy_channels if ch in raw.ch_names]
                    if noisy_channels:
                        logger.info(f"Noisy channels detected: {noisy_channels}")
                        for ch in noisy_channels:
                            if ch not in raw.info['bads'] and ch in raw.ch_names:
                                raw.info['bads'].append(ch)
        except Exception as e:
            logger.warning(f"Could not perform correlation-based bad channel detection: {e}")
        
        # Remove duplicates
        raw.info['bads'] = list(set(raw.info['bads']))
        
        if raw.info['bads']:
            logger.info(f"Total bad channels marked: {raw.info['bads']}")
            
            # Try to interpolate bad channels
            try:
                if len(raw.info['bads']) < len(raw.ch_names) - 1:  # Need at least 2 good channels
                    logger.info(f"Interpolating {len(raw.info['bads'])} bad channels...")
                    raw.interpolate_bads(reset_bads=True)
                    logger.info("Bad channels interpolated successfully")
                else:
                    logger.warning("Too many bad channels to interpolate, will drop them instead")
                    raw.drop_channels(raw.info['bads'])
            except Exception as e:
                logger.warning(f"Could not interpolate bad channels: {e}")
                # Try dropping the bad channels instead
                try:
                    if raw.info['bads']:
                        raw.drop_channels(raw.info['bads'])
                        logger.info("Dropped bad channels instead")
                except:
                    logger.warning("Could not drop bad channels, continuing with all channels")
                    raw.info['bads'] = []
        else:
            logger.info("No bad channels detected")
    
    # Apply basic preprocessing
    logger.info("\nApplying basic preprocessing...")
    
    # Check if we have any channels left
    if len(raw.ch_names) == 0:
        raise ValueError("No channels remaining after filtering non-EEG channels")
    
    try:
        # Apply bandpass filter (0.5-45 Hz for sleep staging)
        logger.info("Applying bandpass filter (0.5-45 Hz)...")
        raw.filter(0.5, 45, fir_design='firwin', verbose=False)
    except Exception as e:
        logger.warning(f"Could not apply bandpass filter: {e}")
        logger.info("Continuing without filtering")
    
    # Get the preprocessed data
    eeg_data = raw.get_data()
    n_channels = eeg_data.shape[0]
    channel_names = raw.ch_names
    
    logger.info(f"Preprocessed data shape: {eeg_data.shape}")
    logger.info(f"Final number of channels: {n_channels}")
    
    if n_channels == 0:
        raise ValueError("No valid EEG channels found in the file")
    
    # Load sleep staging information
    logger.info(f"\nLoading sleep stages from: {os.path.basename(mat_file)}")
    try:
        mat_contents = loadmat(mat_file)
        keys = [key for key in mat_contents.keys() if not key.startswith('__')]
        if len(keys) == 0:
            raise ValueError("No valid variables found in the MAT file")
        
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
        stages_int = [stage_mapping.get(s, 3) for s in stages_clean]  # Default to Wake
        n_classes = len(set(stages_int))
        
        logger.info(f"Loaded {len(stages_int)} sleep stage labels")
        logger.info(f"Number of classes: {n_classes}")
        
        # Print stage distribution
        unique, counts = np.unique(stages_int, return_counts=True)
        stage_names = ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
        logger.info("Stage distribution:")
        for stage_id, count in zip(unique, counts):
            logger.info(f"  {stage_names[stage_id]}: {count} epochs ({count/len(stages_int)*100:.1f}%)")
            
    except Exception as e:
        logger.error(f"Error loading sleep stages: {e}")
        raise
    
    # Create epochs (30-second windows)
    logger.info("\nCreating 30-second epochs...")
    epoch_duration = 30  # seconds
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
            logger.debug(f"Skipping epoch {epoch_idx}: contains NaN or Inf")
            skipped_epochs += 1
            continue
        
        # Check for reasonable amplitude (not flat, not extreme)
        segment_std = np.std(segment)
        if segment_std < 1e-6 or segment_std > 1e4:
            logger.debug(f"Skipping epoch {epoch_idx}: abnormal variance")
            skipped_epochs += 1
            continue
        
        stage = stages_int[epoch_idx]
        segment = segment.astype(np.float32)
        trials.append((segment, stage))
        valid_epochs += 1
    
    logger.info(f"Created {valid_epochs} valid epochs")
    if skipped_epochs > 0:
        logger.info(f"Skipped {skipped_epochs} epochs due to data quality issues")
    
    # Create dataset
    dataset = EEGDataset(trials)
    
    # Compute global normalization statistics
    logger.info("\nComputing normalization statistics...")
    all_segments = np.stack([seg for seg, _ in trials], axis=0)
    global_mean = np.mean(all_segments, axis=(0, 2)).astype(np.float32)
    global_std = np.std(all_segments, axis=(0, 2)).astype(np.float32)
    
    # Prevent division by zero
    global_std = np.where(global_std < 1e-6, 1.0, global_std)
    
    logger.info(f"Global mean shape: {global_mean.shape}")
    logger.info(f"Global std shape: {global_std.shape}")
    
    return dataset, n_channels, n_classes, sfreq, global_mean, global_std


class SleepStagingComparison:
    """Main comparison class for all three SNN approaches."""
    
    def __init__(self, save_dir: str = "comparison_results"):
        """Initialize comparison module."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories
        self.model_dir = os.path.join(save_dir, "models")
        self.plot_dir = os.path.join(save_dir, "plots")
        self.report_dir = os.path.join(save_dir, "reports")
        
        for dir_path in [self.model_dir, self.plot_dir, self.report_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        self.results = {}
        self.models = {}
        self.histories = {}
        self.class_names = ["NREM 1", "NREM 2", "NREM 3", "Wake", "REM"]
        
    def run_comparison(self, dataset, train_indices, eval_indices, 
                      global_mean, global_std, n_channels, n_classes, sfreq,
                      custom_configs: Optional[Dict] = None):
        """Run comparison of all three approaches."""
        
        # Default configurations
        default_configs = self._get_default_configs()
        
        if custom_configs:
            for key in custom_configs:
                if key in default_configs:
                    default_configs[key].update(custom_configs[key])
        
        # Run each option
        logger.info("="*70)
        logger.info("Starting Sleep Staging Model Comparison")
        logger.info("="*70)
        
        # Option A: Pure Population Encoding
        logger.info("\n" + "="*70)
        logger.info("OPTION A: Pure Population-Encoded SNN")
        logger.info("="*70)
        try:
            self._run_option_a(dataset, train_indices, eval_indices, sfreq, 
                             default_configs['option_a'])
        except Exception as e:
            logger.error(f"Option A failed: {str(e)}")
            self.results['option_a'] = {'error': str(e)}
        
        # Option B: Hybrid Model
        logger.info("\n" + "="*70)
        logger.info("OPTION B: Hybrid CNN + Population Encoding")
        logger.info("="*70)
        try:
            self._run_option_b(dataset, train_indices, eval_indices, 
                             global_mean, global_std, n_channels, sfreq,
                             default_configs['option_b'])
        except Exception as e:
            logger.error(f"Option B failed: {str(e)}")
            self.results['option_b'] = {'error': str(e)}
        
        # Option C: Original Model
        logger.info("\n" + "="*70)
        logger.info("OPTION C: Original CNN-Reservoir Model")
        logger.info("="*70)
        try:
            self._run_option_c(dataset, train_indices, eval_indices,
                             global_mean, global_std, n_channels, n_classes, sfreq,
                             default_configs['option_c'], use_enhanced=False)
        except Exception as e:
            logger.error(f"Option C failed: {str(e)}")
            self.results['option_c'] = {'error': str(e)}
        
        # Option C Enhanced: With Attention
        logger.info("\n" + "="*70)
        logger.info("OPTION C-Enhanced: CNN-Reservoir with Attention")
        logger.info("="*70)
        try:
            self._run_option_c(dataset, train_indices, eval_indices,
                             global_mean, global_std, n_channels, n_classes, sfreq,
                             default_configs['option_c_enhanced'], use_enhanced=True)
        except Exception as e:
            logger.error(f"Option C Enhanced failed: {str(e)}")
            self.results['option_c_enhanced'] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_comparison_report()
        
        # Create visualizations
        self._create_comparison_visualizations()
        
        # Save all results
        self._save_results()
        
        return self.results
    
    def _get_default_configs(self) -> Dict:
        """Get default configurations for each option."""
        return {
            'option_a': {
                'n_neurons_per_feature': 50,
                'timesteps': 100,
                'sigma': 0.1,
                'sparse_factor': 0.7,
                'hidden_sizes': [256, 128, 64],
                'n_classes': 5,
                'batch_size': 32,
                'n_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'class_names': self.class_names
            },
            'option_b': {
                'n_neurons_per_feature': 40,
                'timesteps': 50,
                'sigma': 0.15,
                'sparse_factor': 0.8,
                'n_reservoir': 512,
                'n_classes': 5,
                'batch_size': 16,
                'n_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'class_names': self.class_names
            },
            'option_c': {
                'n_reservoir': 512,
                'tau': 0.02,
                'threshold': 1.0,
                'batch_size': 32,
                'n_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'class_names': self.class_names
            },
            'option_c_enhanced': {
                'n_reservoir': 512,
                'tau': 0.02,
                'threshold': 1.0,
                'batch_size': 32,
                'n_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'class_names': self.class_names
            }
        }
    
    def _run_option_a(self, dataset, train_indices, eval_indices, sfreq, config):
        """Run Option A: Pure Population Encoding."""
        start_time = datetime.now()
        
        pipeline = PurePopulationPipeline()
        model, history, results = pipeline.run(
            dataset, train_indices, eval_indices, sfreq, config
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Store results
        self.models['option_a'] = model
        self.histories['option_a'] = history
        self.results['option_a'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'predictions': results['predictions'],
            'labels': results['labels'],
            'training_time': training_time,
            'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'config': config
        }
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(self.model_dir, 'option_a_model.pth'))
    
    def _run_option_b(self, dataset, train_indices, eval_indices, 
                     global_mean, global_std, n_channels, sfreq, config):
        """Run Option B: Hybrid Model."""
        start_time = datetime.now()
        
        pipeline = HybridPipeline()
        model, history, results = pipeline.run(
            dataset, train_indices, eval_indices, global_mean, global_std,
            n_channels, sfreq, config
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Store results
        self.models['option_b'] = model
        self.histories['option_b'] = history
        self.results['option_b'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'predictions': results['predictions'],
            'labels': results['labels'],
            'training_time': training_time,
            'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'config': config
        }
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(self.model_dir, 'option_b_model.pth'))
    
    def _run_option_c(self, dataset, train_indices, eval_indices,
                     global_mean, global_std, n_channels, n_classes, sfreq,
                     config, use_enhanced=False):
        """Run Option C: Original or Enhanced Model."""
        start_time = datetime.now()
        
        pipeline = OriginalModelPipeline()
        model, history, results = pipeline.run(
            dataset, train_indices, eval_indices, global_mean, global_std,
            n_channels, n_classes, sfreq, config, use_enhanced
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Store results
        option_name = 'option_c_enhanced' if use_enhanced else 'option_c'
        self.models[option_name] = model
        self.histories[option_name] = history
        self.results[option_name] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'predictions': results['predictions'],
            'labels': results['labels'],
            'training_time': training_time,
            'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'config': config
        }
        
        # Save model
        model_filename = f'{option_name}_model.pth'
        torch.save(model.state_dict(), 
                  os.path.join(self.model_dir, model_filename))
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        report = []
        report.append("="*80)
        report.append("SLEEP STAGING MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        summary_data = []
        headers = ["Model", "Accuracy (%)", "F1 Score", "Training Time (s)", 
                   "Parameters", "Status"]
        
        model_names = {
            'option_a': 'Pure Population SNN',
            'option_b': 'Hybrid CNN+Population',
            'option_c': 'Original CNN-Reservoir',
            'option_c_enhanced': 'Enhanced CNN-Attention'
        }
        
        for key, name in model_names.items():
            if key in self.results:
                result = self.results[key]
                if 'error' in result:
                    summary_data.append([name, "-", "-", "-", "-", "Failed"])
                else:
                    summary_data.append([
                        name,
                        f"{result['accuracy']:.2f}",
                        f"{result['f1_score']:.4f}",
                        f"{result['training_time']:.1f}",
                        f"{result['model_params']:,}",
                        "Success"
                    ])
        
        report.append("\nSUMMARY")
        report.append("-"*80)
        report.append(tabulate(summary_data, headers=headers, tablefmt="grid"))
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.report_dir, "comparison_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
    
    def _create_comparison_visualizations(self):
        """Create comparison visualizations."""
        # Overall performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = []
        accuracies = []
        f1_scores = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (key, name) in enumerate([
            ('option_a', 'Pure\nPopulation'),
            ('option_b', 'Hybrid\nCNN+Pop'),
            ('option_c', 'Original\nCNN-Res'),
            ('option_c_enhanced', 'Enhanced\nCNN-Att')
        ]):
            if key in self.results and 'error' not in self.results[key]:
                model_names.append(name)
                accuracies.append(self.results[key]['accuracy'])
                f1_scores.append(self.results[key]['f1_score'])
        
        if model_names:
            # Accuracy comparison
            x = np.arange(len(model_names))
            bars1 = ax1.bar(x, accuracies, color=colors[:len(model_names)], alpha=0.8)
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names)
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # F1 Score comparison
            bars2 = ax2.bar(x, f1_scores, color=colors[:len(model_names)], alpha=0.8)
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_names)
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, f1 in zip(bars2, f1_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'overall_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save all results to disk."""
        results_path = os.path.join(self.save_dir, 'comparison_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'histories': self.histories,
                'class_names': self.class_names
            }, f)
        
        logger.info(f"All results saved to {self.save_dir}")
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """Get the best performing model based on F1 score."""
        best_model = None
        best_f1 = 0
        
        for key in self.results:
            if 'error' not in self.results[key]:
                if self.results[key]['f1_score'] > best_f1:
                    best_f1 = self.results[key]['f1_score']
                    best_model = key
        
        if best_model:
            return best_model, self.results[best_model]
        else:
            return None, None


def main():
    """Main function to run the complete sleep staging comparison."""
    
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
    
    try:
        # Load and preprocess data with MNE bad channel detection
        dataset, n_channels, n_classes, sfreq, global_mean, global_std = load_and_preprocess_data(
            edf_file, mat_file, use_mne_cleaning=True
        )
        
        # Split dataset
        train_split = int(len(dataset) * 0.8)
        train_indices = list(range(train_split))
        eval_indices = list(range(train_split, len(dataset)))
        
        logger.info(f"\nDataset split:")
        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Evaluation samples: {len(eval_indices)}")
        
        # Run comparison
        comparison = SleepStagingComparison(save_dir="sleep_staging_results")
        
        # Optional: Custom configurations for faster testing
        custom_configs = {
            'option_a': {'n_epochs': 20, 'batch_size': 32},
            'option_b': {'n_epochs': 20, 'batch_size': 16},
            'option_c': {'n_epochs': 20, 'batch_size': 32},
            'option_c_enhanced': {'n_epochs': 20, 'batch_size': 32}
        }
        
        # Run the comparison
        results = comparison.run_comparison(
            dataset, train_indices, eval_indices,
            global_mean, global_std, n_channels, n_classes, sfreq,
            custom_configs
        )
        
        # Get and display best model
        best_model_key, best_model_results = comparison.get_best_model()
        if best_model_key:
            logger.info("\n" + "="*70)
            logger.info("BEST PERFORMING MODEL")
            logger.info("="*70)
            logger.info(f"Model: {best_model_key}")
            logger.info(f"Accuracy: {best_model_results['accuracy']:.2f}%")
            logger.info(f"F1 Score: {best_model_results['f1_score']:.4f}")
            logger.info(f"Training Time: {best_model_results['training_time']:.1f}s")
            logger.info("="*70)
        
        logger.info("\nComparison completed successfully!")
        logger.info(f"Results saved in: sleep_staging_results/")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()