# NeuroTrack: Alzheimer’s Disease Stage Prediction from MRI

<img width="1500" height="214" alt="image" src="https://github.com/user-attachments/assets/93e17971-ef99-4bac-815b-ee63c731ef21" />


> [!WARNING]
> This project is currently a work in progress.

Deep learning-based system for stage-wise classification of Alzheimer’s disease using brain MRI scans, with a focus on modeling structural patterns from slice-based data.

---

## Overview

Alzheimer’s disease is a progressive neurodegenerative disorder characterized by structural changes in the brain. Early identification of disease stages is critical for understanding progression and enabling timely clinical intervention.

This project explores a data-driven approach to classify stages of Alzheimer’s disease using MRI scans. Given the absence of full 3D volumetric data, the project introduces a slice-based approximation to capture structural information.

---

## Dataset

- Source: Kaggle (Augmented Alzheimer MRI Dataset)

- Dataset Link: https://kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset (Currently Not Accessible)

- Classes:
    
    - Non Demented
        
    - Very Mild Demented
        
    - Mild Demented
        
    - Moderate Demented
        
- Dataset split:
    
    - Train (70%)
        
    - Validation (15%)
        
    - Test (15%)
        

> [!NOTE]
> Training data includes augmented images, while evaluation is performed on primarily non-augmented data to ensure unbiased assessment.

---

## Methodology

### 1. Data Representation

- MRI data is available as 2D slices rather than full 3D volumes.
    
- To approximate spatial structure, consecutive slices are grouped.
    

### 2. 2.5D Input Modeling

- Three consecutive MRI slices are treated as a 3-channel input:  
	```
    [slice_i, slice_{i+1}, slice_{i+2}]
    ```
- This enables the model to capture local inter-slice continuity.
    

### 3. Model Architecture

- Custom CNN architecture
    
- Transfer learning models (e.g., ResNet)
    
- Multi-class classification setup
    

### 4. Training

- Cross-entropy loss
    
- Standard image preprocessing and normalization
    
- Validation-based performance monitoring
    

---

## Inference Strategy

### Multi-Slice Prediction

- Input can consist of multiple consecutive slices.
    
- Slices are grouped into sets of three using a sliding window approach.
    

### Aggregation

- Predictions from multiple slice groups are combined using:
    
    - Majority voting
        
    - Probability averaging
        

This enables patient-level prediction rather than relying on a single slice.

---

## Robustness

- Test-time augmentation is applied during inference.
    
- Input images are slightly transformed and re-evaluated multiple times.
    
- Final predictions are aggregated to improve reliability.
    

---

## Results

(To be updated)

---

## Key Insight

This project bridges the gap between 2D slice-based datasets and 3D structural understanding by using a 2.5D approach, enabling the model to learn local anatomical patterns associated with neurodegeneration.

---

## Tech Stack

- Python
    
- PyTorch
    
- torchvision
    
- OpenCV
    
- NumPy
    
- Matplotlib
    

---

## Future Work

- Incorporate longitudinal data for progression prediction
    
- Use full 3D MRI datasets for volumetric modeling
    
- Deploy model via web interface for real-time inference
    

---

## Author

Jivesh Gawde & Mahat Vasudev
