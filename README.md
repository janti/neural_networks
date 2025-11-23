"# Tree Classification with Deep Learning

A comprehensive machine learning project implementing transfer learning with ResNet architectures for automated tree species identification from urban street images.

## 🌳 Project Overview

This project demonstrates the application of convolutional neural networks (CNNs) for multi-class image classification, specifically focusing on identifying 23 different tree species from photographs. Using PyTorch and transfer learning techniques, multiple ResNet architectures were trained and evaluated to achieve optimal performance.

### Key Achievements
- **89.19% accuracy** with ResNet50 fine-tuning approach
- **6 different model configurations** tested and compared
- **Comprehensive evaluation** including n-best classification analysis
- **Robust data preprocessing** with effective augmentation strategies

## 📊 Dataset

**Tree Dataset of Urban Street Classification** (Kaggle)
- **23 tree species** including:
  - Acer palmatum, Ginkgo biloba, Magnolia grandiflora
  - Cedrus deodara, Flowering cherry, Platanus
  - And 17 additional species
- **4,804 total images** distributed across:
  - Training: 3,850 images
  - Validation: 482 images
  - Test: 472 images

## 🏗️ Model Architectures

### Tested Configurations

1. **ResNet18** (Fine-tuning) - 84.96% accuracy
2. **ResNet50** (Fine-tuning) - **89.62% accuracy** ⭐
3. **ResNet18** (Feature extraction) - 56.14% accuracy
4. **ResNet50** (Feature extraction) - 54.45% accuracy
5. **ResNet18** (High learning rate) - Various configurations
6. **ResNet50** (Extended training) - Long-term optimization

### Best Performing Model
- **Architecture**: ResNet50 with fine-tuning
- **Test Accuracy**: 89.62%
- **Precision**: 91.08%
- **Recall**: 89.62%
- **F1-Score**: 89.70%

## 📈 Performance Metrics

### N-Best Classification Results (ResNet50)
- **Top-1**: 89.62%
- **Top-2**: 94.92%
- **Top-3**: 96.40%
- **Top-4**: 97.88%
- **Top-5**: 98.31%

## 🛠️ Technical Implementation

### Data Preprocessing
- **RandomResizedCrop** (224×224) for spatial variability
- **RandomHorizontalFlip** for data augmentation
- **ImageNet normalization** for transfer learning compatibility

### Training Strategy
- **Transfer Learning** with pre-trained ImageNet weights
- **Adam optimizer** with learning rate scheduling
- **Cross-entropy loss** for multi-class classification
- **Early stopping** to prevent overfitting

## 📁 Project Structure

```
neural_networks/
├── README.md                           # Project documentation
├── Tree_Classification_Report.md       # Detailed technical report
├── results_summary.txt                 # Performance summary
├── exercise_release.ipynb             # Main notebook
├── exercise_release – kopio.ipynb     # Notebook copy
├── tree_dataset/                      # Dataset directory
│   ├── train/                         # Training images (23 classes)
│   ├── val/                           # Validation images
│   └── test/                          # Test images
└── *.pth                              # Trained model weights
    ├── resnet18_tree_classifier.pth
    ├── resnet50_tree_classifier.pth
    ├── resnet18_feature_extract_tree_classifier.pth
    ├── resnet50_feature_extract_tree_classifier.pth
    ├── resnet18_high_lr_tree_classifier.pth
    └── resnet50_feature_extract_long_tree_classifier.pth
```

## 🚀 Getting Started

### Prerequisites
```bash
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
PIL
sklearn
```

### Running the Project
1. Clone the repository
2. Ensure the `tree_dataset/` directory contains the properly structured data
3. Open `exercise_release.ipynb` in Jupyter Notebook
4. Run cells sequentially to train and evaluate models

### Model Loading
```python
import torch
model = torch.load('resnet50_tree_classifier.pth')
model.eval()
```

## 📋 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| ResNet18 (Fine-tune) | 84.96% | 86.44% | 84.96% | 84.62% |
| **ResNet50 (Fine-tune)** | **89.62%** | **91.08%** | **89.62%** | **89.70%** |
| ResNet18 (Feature Extract) | 56.14% | 56.38% | 56.14% | 54.28% |
| ResNet50 (Feature Extract) | 54.45% | 56.49% | 54.45% | 53.05% |

## 🔍 Key Findings

1. **Fine-tuning outperforms feature extraction** significantly
2. **ResNet50 superior to ResNet18** for this dataset
3. **Data augmentation crucial** for generalization
4. **Transfer learning highly effective** for limited dataset sizes
5. **N-best classification** shows excellent top-5 performance (98.31%)

## 📖 Documentation

For detailed methodology, experimental setup, and comprehensive analysis, see:
- `Tree_Classification_Report.md` - Complete technical report
- `results_summary.txt` - Raw performance metrics
- `exercise_release.ipynb` - Implementation notebook

## 👨‍💻 Author

**Jani Timmerheid**  
Neural Networks Course Project  
November 2025

## 📄 License

This project is for educational purposes as part of a Neural Networks course.

---

*This project demonstrates practical application of deep learning techniques for real-world image classification challenges, showcasing the effectiveness of transfer learning in computer vision tasks.*" 
