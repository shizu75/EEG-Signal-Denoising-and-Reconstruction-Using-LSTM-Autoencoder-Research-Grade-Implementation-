# EEG Signal Denoising and Reconstruction Using LSTM Autoencoder

## Overview
This repository presents a **research-grade deep learning pipeline** for **EEG signal denoising and reconstruction** using a **sequence-to-sequence LSTM Autoencoder**. The framework is designed to suppress noise and artifacts in EEG recordings while preserving temporal and physiological characteristics of neural signals. The methodology aligns with standards used in **biomedical signal processing, brain–computer interface (BCI), and neuroengineering research**.

---

## Dataset Description
- **Primary Dataset:** `EEG_data_ICA1.csv`
- **Test Dataset:** `testdatafordataset.csv`
- **Additional Validation Dataset:** `FilteredICA_datasetfortrain&test.csv`
- **Sampling Rates Used:**
  - 500 Hz (raw EEG)
  - 128 Hz (ICA-filtered EEG)
- **Data Format:** Each row corresponds to a single EEG trial; columns represent time samples.

---

## Signal Visualization
- Time-domain EEG signals are plotted for:
  - Raw EEG
  - ICA-filtered EEG
- Mean signal comparison between raw and filtered datasets to validate preprocessing effectiveness.

---

## Data Augmentation
To improve generalization and robustness:
- **Time-shift augmentation** is applied with random temporal offsets.
- Each EEG signal is augmented **18×**.
- Augmentation strategy simulates realistic temporal variability without altering signal morphology.

---

## Preprocessing Pipeline
1. **Min–Max Scaling** to range `[-1, 1]`
2. **Sliding Window Segmentation**
   - Window size: `10`
   - Input–target pairs constructed for sequence reconstruction
3. **Dataset Reshaping**
   - Final shape: `(samples, timesteps, channels)`
   - Channel dimension = 1

---

## Train–Validation–Test Split
- Training set
- Validation set
- Test set
- Split strategy ensures unbiased generalization evaluation.

---

## Model Architecture: LSTM Autoencoder
The denoising model is a **deep recurrent autoencoder** composed of:

### Encoder
- LSTM (64 units, ReLU, return sequences)
- LSTM (32 units, ReLU)

### Bottleneck
- RepeatVector (sequence length = 10)

### Decoder
- LSTM (32 units, ReLU, return sequences)
- LSTM (64 units, ReLU, return sequences)
- TimeDistributed Dense (1 neuron)

This architecture captures both **short- and long-term temporal dependencies** in EEG signals.

---

## Training Configuration
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 32
- **Epochs:** Up to 100
- **Callbacks:**
  - EarlyStopping (patience = 8, restore best weights)
  - ReduceLROnPlateau (adaptive learning rate decay)

---

## Performance Monitoring
- Training and validation loss curves plotted across epochs
- Overfitting controlled via early stopping
- Final evaluation performed on unseen test data

---

## Reconstruction Analysis
- Inverse scaling applied to reconstructed signals
- Comparative plots include:
  - Original EEG
  - Reconstructed EEG
  - Reconstruction error (shaded area)
- Time-domain plots demonstrate effective noise suppression while preserving waveform integrity.

---

## Model Persistence
- Trained model saved as: `eeg_denoiser.keras`
- Reloaded successfully for inference and validation
- Supports deployment and reproducibility.

---

## Inference on Unseen EEG Data
- Raw EEG reshaped and normalized
- Passed through trained autoencoder
- Reconstructed signal inverse-transformed
- Visual and numerical validation performed on:
  - `testdatafordataset.csv`
  - `testdata.csv`

---

## Key Contributions
- Robust EEG denoising using deep recurrent learning
- Data augmentation tailored for biomedical time-series
- Fully reproducible preprocessing–training–evaluation pipeline
- Applicable to:
  - EEG artifact removal
  - BCI preprocessing
  - Neural signal enhancement
  - Clinical and cognitive neuroscience research

---

## Technologies Used
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## Research Relevance
This implementation meets **graduate-level and PhD-level research standards**, emphasizing:
- Signal integrity
- Temporal modeling rigor
- Clear experimental validation
- Extendability to multi-channel EEG and clinical datasets

---

## Potential Extensions
- Multichannel EEG modeling
- Frequency-domain loss integration
- Attention-based sequence autoencoders
- Real-time EEG denoising systems
- Clinical artifact classification

---

## License
For academic and research use only.
