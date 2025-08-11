"""
Data Augmentation Techniques for In-Ear EEG
===========================================
Specialized augmentation methods that preserve sleep stage characteristics
while increasing dataset diversity.
"""

import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class InEarEEGAugmentation:
    """
    Comprehensive augmentation techniques for in-ear EEG sleep staging.
    These methods preserve the physiological characteristics of sleep stages.
    """
    
    def __init__(self, sfreq=200.0):
        self.sfreq = sfreq
        
    # ============================================
    # 1. AMPLITUDE-BASED AUGMENTATIONS
    # ============================================
    
    def amplitude_scaling(self, eeg_signal, scale_range=(0.8, 1.2)):
        """
        Random amplitude scaling - simulates inter-subject variability.
        Sleep stages are characterized by relative, not absolute amplitudes.
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return eeg_signal * scale
    
    def channel_wise_scaling(self, eeg_signal):
        """
        Different scaling per channel - simulates electrode impedance variations.
        Common in in-ear EEG due to ear canal differences.
        """
        if len(eeg_signal.shape) == 2:
            n_channels = eeg_signal.shape[0]
            scales = np.random.uniform(0.85, 1.15, (n_channels, 1))
            return eeg_signal * scales
        return eeg_signal
    
    def dc_shift(self, eeg_signal, shift_range=(-10, 10)):
        """
        Add random DC offset - simulates baseline drift.
        Common in long recordings due to electrode drift.
        """
        shift = np.random.uniform(shift_range[0], shift_range[1])
        return eeg_signal + shift
    
    # ============================================
    # 2. NOISE AUGMENTATIONS
    # ============================================
    
    def add_gaussian_noise(self, eeg_signal, snr_db=40):
        """
        Add Gaussian noise at specified SNR.
        Simulates environmental and biological noise.
        """
        signal_power = np.mean(eeg_signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), eeg_signal.shape)
        return eeg_signal + noise
    
    def add_pink_noise(self, eeg_signal, amplitude=0.01):
        """
        Add 1/f (pink) noise - more realistic for biological signals.
        Pink noise is common in EEG recordings.
        """
        if len(eeg_signal.shape) == 2:
            n_channels, n_samples = eeg_signal.shape
        else:
            n_channels = 1
            n_samples = len(eeg_signal)
            eeg_signal = eeg_signal.reshape(1, -1)
        
        # Generate pink noise
        pink_noise = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Create white noise
            white = np.random.randn(n_samples)
            # Apply 1/f filter to create pink noise
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_samples, 1/self.sfreq)
            freqs[0] = 1  # Avoid division by zero
            fft = fft / np.sqrt(freqs)
            pink_noise[ch] = np.fft.irfft(fft, n_samples) * amplitude
        
        augmented = eeg_signal + pink_noise * np.std(eeg_signal)
        return augmented.squeeze() if n_channels == 1 else augmented
    
    def add_movement_artifact(self, eeg_signal, probability=0.1, duration_sec=0.5):
        """
        Simulate movement artifacts - common in sleep recordings.
        Brief high-amplitude disturbances.
        """
        if np.random.random() > probability:
            return eeg_signal
        
        n_samples = eeg_signal.shape[-1]
        artifact_duration = int(duration_sec * self.sfreq)
        artifact_start = np.random.randint(0, max(1, n_samples - artifact_duration))
        
        # Create artifact (high amplitude, low frequency)
        t = np.arange(artifact_duration) / self.sfreq
        artifact = np.sin(2 * np.pi * np.random.uniform(0.5, 2) * t) * np.std(eeg_signal) * 5
        
        augmented = eeg_signal.copy()
        if len(augmented.shape) == 2:
            augmented[:, artifact_start:artifact_start + artifact_duration] += artifact
        else:
            augmented[artifact_start:artifact_start + artifact_duration] += artifact
        
        return augmented
    
    # ============================================
    # 3. TEMPORAL AUGMENTATIONS
    # ============================================
    
    def time_stretching(self, eeg_signal, stretch_factor_range=(0.95, 1.05)):
        """
        Slight time stretching - simulates heart rate variability effects.
        Preserves frequency content while changing temporal dynamics.
        """
        stretch_factor = np.random.uniform(stretch_factor_range[0], stretch_factor_range[1])
        
        if len(eeg_signal.shape) == 2:
            n_channels, n_samples = eeg_signal.shape
            new_length = int(n_samples * stretch_factor)
            stretched = np.zeros((n_channels, n_samples))
            
            for ch in range(n_channels):
                # Interpolate to new length
                old_indices = np.arange(n_samples)
                new_indices = np.linspace(0, n_samples - 1, new_length)
                f = interp1d(new_indices, eeg_signal[ch, :new_length], 
                           kind='linear', fill_value='extrapolate')
                stretched[ch] = f(old_indices)
        else:
            n_samples = len(eeg_signal)
            new_length = int(n_samples * stretch_factor)
            old_indices = np.arange(n_samples)
            new_indices = np.linspace(0, n_samples - 1, new_length)
            f = interp1d(new_indices, eeg_signal[:new_length], 
                       kind='linear', fill_value='extrapolate')
            stretched = f(old_indices)
        
        return stretched
    
    def random_crop_and_pad(self, eeg_signal, crop_size=0.9):
        """
        Random cropping with padding - creates temporal shifts.
        Helps model be robust to epoch boundary variations.
        """
        n_samples = eeg_signal.shape[-1]
        crop_samples = int(n_samples * crop_size)
        start = np.random.randint(0, n_samples - crop_samples)
        
        if len(eeg_signal.shape) == 2:
            cropped = eeg_signal[:, start:start + crop_samples]
            # Pad with edge values
            pad_left = start
            pad_right = n_samples - crop_samples - start
            padded = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode='edge')
        else:
            cropped = eeg_signal[start:start + crop_samples]
            pad_left = start
            pad_right = n_samples - crop_samples - start
            padded = np.pad(cropped, (pad_left, pad_right), mode='edge')
        
        return padded
    
    # ============================================
    # 4. FREQUENCY DOMAIN AUGMENTATIONS
    # ============================================
    
    def frequency_masking(self, eeg_signal, max_mask_freq=5, n_masks=1):
        """
        Mask random frequency bands - makes model robust to missing frequencies.
        Useful for handling individual differences in spectral peaks.
        """
        augmented = eeg_signal.copy()
        
        for _ in range(n_masks):
            if len(augmented.shape) == 2:
                for ch in range(augmented.shape[0]):
                    fft = np.fft.rfft(augmented[ch])
                    freqs = np.fft.rfftfreq(augmented.shape[1], 1/self.sfreq)
                    
                    # Random frequency band to mask
                    mask_start = np.random.uniform(0, 30)
                    mask_width = np.random.uniform(1, max_mask_freq)
                    mask_idx = np.logical_and(freqs >= mask_start, 
                                            freqs <= mask_start + mask_width)
                    fft[mask_idx] = 0
                    augmented[ch] = np.fft.irfft(fft, augmented.shape[1])
            else:
                fft = np.fft.rfft(augmented)
                freqs = np.fft.rfftfreq(len(augmented), 1/self.sfreq)
                mask_start = np.random.uniform(0, 30)
                mask_width = np.random.uniform(1, max_mask_freq)
                mask_idx = np.logical_and(freqs >= mask_start, 
                                        freqs <= mask_start + mask_width)
                fft[mask_idx] = 0
                augmented = np.fft.irfft(fft, len(augmented))
        
        return augmented
    
    def band_stop_filter(self, eeg_signal, stop_band=None):
        """
        Apply random band-stop filter - simulates frequency-specific artifacts.
        Common for powerline noise or specific interference.
        """
        if stop_band is None:
            # Random band between 5-40 Hz
            center_freq = np.random.uniform(5, 40)
            bandwidth = np.random.uniform(1, 5)
            stop_band = (center_freq - bandwidth/2, center_freq + bandwidth/2)
        
        # Design butterworth bandstop filter
        nyquist = self.sfreq / 2
        low = stop_band[0] / nyquist
        high = stop_band[1] / nyquist
        
        if low > 0 and high < 1:
            b, a = signal.butter(2, [low, high], btype='bandstop')
            if len(eeg_signal.shape) == 2:
                filtered = np.zeros_like(eeg_signal)
                for ch in range(eeg_signal.shape[0]):
                    filtered[ch] = signal.filtfilt(b, a, eeg_signal[ch])
                return filtered
            else:
                return signal.filtfilt(b, a, eeg_signal)
        
        return eeg_signal
    
    # ============================================
    # 5. SLEEP-SPECIFIC AUGMENTATIONS
    # ============================================
    
    def simulate_k_complex(self, eeg_signal, probability=0.05):
        """
        Add synthetic K-complexes - characteristic of NREM2.
        Only apply to NREM2 epochs for realism.
        """
        if np.random.random() > probability:
            return eeg_signal
        
        n_samples = eeg_signal.shape[-1]
        # K-complex: sharp negative then positive wave, ~0.5-1.5 seconds
        duration = int(np.random.uniform(0.5, 1.5) * self.sfreq)
        position = np.random.randint(duration, n_samples - duration)
        
        # Create K-complex shape
        t = np.arange(duration) / self.sfreq
        k_complex = -np.sin(2 * np.pi * 0.5 * t) * np.exp(-t * 2)
        k_complex *= np.std(eeg_signal) * np.random.uniform(2, 4)
        
        augmented = eeg_signal.copy()
        if len(augmented.shape) == 2:
            # Add to all channels with slight variations
            for ch in range(augmented.shape[0]):
                ch_variation = k_complex * np.random.uniform(0.8, 1.2)
                augmented[ch, position:position + duration] += ch_variation
        else:
            augmented[position:position + duration] += k_complex
        
        return augmented
    
    def simulate_sleep_spindle(self, eeg_signal, probability=0.05):
        """
        Add synthetic sleep spindles - characteristic of NREM2.
        11-15 Hz oscillations lasting 0.5-2 seconds.
        """
        if np.random.random() > probability:
            return eeg_signal
        
        n_samples = eeg_signal.shape[-1]
        # Spindle parameters
        spindle_freq = np.random.uniform(11, 15)  # Hz
        duration = np.random.uniform(0.5, 2.0)  # seconds
        duration_samples = int(duration * self.sfreq)
        position = np.random.randint(0, max(1, n_samples - duration_samples))
        
        # Create spindle
        t = np.arange(duration_samples) / self.sfreq
        envelope = np.sin(np.pi * t / duration) ** 2  # Spindle envelope
        spindle = np.sin(2 * np.pi * spindle_freq * t) * envelope
        spindle *= np.std(eeg_signal) * np.random.uniform(1, 2)
        
        augmented = eeg_signal.copy()
        if len(augmented.shape) == 2:
            for ch in range(augmented.shape[0]):
                ch_spindle = spindle * np.random.uniform(0.8, 1.2)
                augmented[ch, position:position + duration_samples] += ch_spindle
        else:
            augmented[position:position + duration_samples] += spindle
        
        return augmented
    
    # ============================================
    # 6. MIXUP AUGMENTATION
    # ============================================
    
    def mixup(self, eeg1, eeg2, label1, label2, alpha=0.2):
        """
        Mixup augmentation - blend two epochs.
        Particularly effective for sleep staging.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed_eeg = lam * eeg1 + (1 - lam) * eeg2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_eeg, mixed_label
    
    # ============================================
    # 7. COMPOSITE AUGMENTATION
    # ============================================
    
    def augment_epoch(self, eeg_signal, augmentation_probability=0.8):
        """
        Apply multiple augmentations with controlled probability.
        This is the main function to use during training.
        """
        if np.random.random() > augmentation_probability:
            return eeg_signal
        
        augmented = eeg_signal.copy()
        
        # Select which augmentations to apply
        augmentations = []
        
        # Always apply some basic augmentations
        if np.random.random() < 0.8:
            augmentations.append(('amplitude_scaling', {}))
        
        if np.random.random() < 0.5:
            augmentations.append(('add_gaussian_noise', {'snr_db': np.random.uniform(30, 50)}))
        
        if np.random.random() < 0.3:
            augmentations.append(('add_pink_noise', {'amplitude': np.random.uniform(0.005, 0.02)}))
        
        if np.random.random() < 0.2:
            augmentations.append(('time_stretching', {}))
        
        if np.random.random() < 0.1:
            augmentations.append(('add_movement_artifact', {}))
        
        if np.random.random() < 0.2:
            augmentations.append(('frequency_masking', {}))
        
        # Apply selected augmentations
        for aug_name, params in augmentations:
            aug_func = getattr(self, aug_name)
            augmented = aug_func(augmented, **params)
        
        return augmented


class AugmentedEEGDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset with on-the-fly augmentation for in-ear EEG.
    """
    
    def __init__(self, trials, labels, sfreq=200, augment=True, 
                 augmentation_prob=0.8, mixup_alpha=0.2):
        self.trials = trials
        self.labels = labels
        self.sfreq = sfreq
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.mixup_alpha = mixup_alpha
        self.augmenter = InEarEEGAugmentation(sfreq)
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        eeg = self.trials[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            # Standard augmentations
            eeg = self.augmenter.augment_epoch(eeg, self.augmentation_prob)
            
            # Mixup (occasionally)
            if np.random.random() < 0.3 and self.mixup_alpha > 0:
                # Get another random sample
                idx2 = np.random.randint(0, len(self.trials))
                eeg2 = self.trials[idx2].copy()
                label2 = self.labels[idx2]
                
                # Apply mixup
                eeg, label = self.augmenter.mixup(
                    eeg, eeg2, label, label2, self.mixup_alpha
                )
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg)
        label_tensor = torch.LongTensor([label])[0] if not isinstance(label, (list, np.ndarray)) else torch.FloatTensor(label)
        
        return eeg_tensor, label_tensor


# ============================================
# USAGE EXAMPLE
# ============================================

def demonstrate_augmentations():
    """
    Demonstrate all augmentation techniques with visualization.
    """
    import matplotlib.pyplot as plt
    
    # Create synthetic in-ear EEG (2 channels, 30 seconds at 200 Hz)
    sfreq = 200
    duration = 30
    n_samples = sfreq * duration
    n_channels = 2
    
    # Simulate sleep EEG with different frequency components
    t = np.arange(n_samples) / sfreq
    eeg = np.zeros((n_channels, n_samples))
    
    # Channel 1: Mix of frequencies typical in sleep
    eeg[0] = (2 * np.sin(2 * np.pi * 1.5 * t) +  # Delta
              1 * np.sin(2 * np.pi * 6 * t) +    # Theta
              0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
              0.3 * np.sin(2 * np.pi * 20 * t))   # Beta
    
    # Channel 2: Similar but with phase shift
    eeg[1] = (2 * np.sin(2 * np.pi * 1.5 * t + np.pi/4) +
              1 * np.sin(2 * np.pi * 6 * t + np.pi/3) +
              0.5 * np.sin(2 * np.pi * 10 * t + np.pi/2) +
              0.3 * np.sin(2 * np.pi * 20 * t + np.pi))
    
    # Add some noise
    eeg += np.random.randn(*eeg.shape) * 0.1
    
    # Initialize augmenter
    augmenter = InEarEEGAugmentation(sfreq)
    
    # Apply different augmentations
    augmentations = {
        'Original': eeg,
        'Amplitude Scaling': augmenter.amplitude_scaling(eeg),
        'Gaussian Noise': augmenter.add_gaussian_noise(eeg, snr_db=30),
        'Pink Noise': augmenter.add_pink_noise(eeg, amplitude=0.02),
        'Movement Artifact': augmenter.add_movement_artifact(eeg, probability=1.0),
        'Time Stretching': augmenter.time_stretching(eeg),
        'Frequency Masking': augmenter.frequency_masking(eeg),
        'K-Complex': augmenter.simulate_k_complex(eeg, probability=1.0),
        'Sleep Spindle': augmenter.simulate_sleep_spindle(eeg, probability=1.0),
        'Combined': augmenter.augment_epoch(eeg, augmentation_probability=1.0)
    }
    
    # Plot comparisons
    fig, axes = plt.subplots(5, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (name, aug_eeg) in enumerate(augmentations.items()):
        ax = axes[idx]
        # Plot first 5 seconds for clarity
        t_plot = t[:1000]
        ax.plot(t_plot, aug_eeg[0, :1000], 'b-', alpha=0.7, label='Ch1')
        ax.plot(t_plot, aug_eeg[1, :1000], 'r-', alpha=0.7, label='Ch2')
        ax.set_title(name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('eeg_augmentations_demo.png', dpi=150)
    plt.show()
    
    print("Augmentation demonstration complete!")
    print("\nKey insights for in-ear EEG augmentation:")
    print("1. Amplitude variations simulate inter-subject differences")
    print("2. Noise additions make the model robust to real-world conditions")
    print("3. Frequency masking handles individual spectral variations")
    print("4. Sleep-specific augmentations (K-complexes, spindles) for NREM2")
    print("5. Mixup is particularly effective for sleep staging")


if __name__ == "__main__":
    demonstrate_augmentations()
