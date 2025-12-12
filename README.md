# data_augmentation_for_CNN
This project demonstrates image data augmentation techniques using TensorFlow/Keras to enhance CNN training on a car classification dataset. It applies random flips, contrast adjustments, and explores pipeline optimizations to reduce overfitting and improve model generalization.

## Project Description

This project demonstrates the practical application of data augmentation techniques to improve model generalization and reduce overfitting in deep learning. Using the **modified cars196 dataset** from Kaggle's Computer Vision course, we implement a full training pipeline that includes image preprocessing, real-time augmentation, model building, training, and evaluation. The notebook is structured to be both educational and reproducible, suitable for learners and practitioners in computer vision.

## Project Structure

```
data_augmentation_CNN/
├── data_augmentation_CNN.ipynb  # Main Jupyter notebook
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
└── requirements.txt              # Dependencies
```

## Key Concepts Covered

- **Data Augmentation in CNNs**: Using random transformations (contrast, flips, rotation) to artificially expand the training dataset
- **TensorFlow Data Pipeline**: Efficient data loading, caching, and prefetching with `tf.data`
- **Model Integration**: Building augmentation directly into the Keras model as layers
- **Comparative Analysis**: Training and evaluating two CNN architectures with different augmentation strategies
- **Performance Visualization**: Plotting training/validation loss and accuracy curves

## Technical Implementation

### 1. **Data Loading & Preprocessing**
- Loads images from directory structure using `image_dataset_from_directory`
- Converts images to `tf.float32` normalized range [0, 1]
- Implements data caching and prefetching for optimal GPU utilization

### 2. **Augmentation Techniques**
The notebook implements several augmentation methods:
- `RandomContrast`: Adjusts image contrast by a specified factor
- `RandomFlip`: Horizontal and vertical image flipping
- `RandomRotation`: Image rotation within specified range
- *Additional techniques are commented for experimentation*

### 3. **CNN Architectures**
**Model 1: Standard CNN with Moderate Augmentation**
- Augmentation: Contrast(0.1), Horizontal Flip, Rotation(0.1)
- Convolutional blocks with 64, 128, 256 filters (kernel_size=3)
- Activation: ReLU, Padding: 'same'
- Dense head: 8 neurons → 1 output (sigmoid)

**Model 2: Alternative CNN with Strong Augmentation**
- Augmentation: Contrast(0.5), Vertical Flip, Rotation(0.5)
- Convolutional blocks with 64, 128, 256 filters (kernel_size=2)
- Activation: SELU, Padding: 'valid'
- Dense head: 6 neurons → 1 output (sigmoid)

### 4. **Training Configuration**
- Optimizer: Adam with epsilon=0.01
- Loss: Binary Crossentropy
- Metric: Binary Accuracy
- Epochs: 40 (with early stopping potential)
- Batch Size: 64

## Results & Evaluation

The notebook includes comprehensive evaluation:
- Training and validation loss plots for both models
- Accuracy progression throughout training
- Best validation loss and accuracy metrics comparison
- Visual examples of augmented images

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/data_augmentation_CNN.git
cd data_augmentation_CNN

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook data_augmentation_CNN.ipynb
```

### Dataset Setup
The notebook uses the Kaggle Cars vs Trucks dataset. You can:
1. Use the provided Kaggle dataset paths in the notebook
2. Replace with your own image dataset by modifying the directory paths
3. Ensure your dataset follows the structure: `train/class1/`, `train/class2/`

## Customization

You can easily modify:
- **Augmentation parameters**: Adjust factor values in augmentation layers
- **Model architecture**: Change filter sizes, activation functions, or add/remove layers
- **Training parameters**: Modify optimizer, learning rate, or number of epochs
- **Dataset**: Replace with your own image classification dataset

## Expected Outcomes

After running the notebook, you will:
1. Understand how data augmentation affects CNN training
2. Have a reusable pipeline for image classification tasks
3. Be able to compare different augmentation strategies
4. Gain insights into model generalization and overfitting prevention

## References

1. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning.
2. Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best practices for convolutional neural networks applied to visual document analysis.
3. LeCun, Y., et al. Comparison of learning algorithms for handwritten digit recognition.

## Contributing

Feel free to fork this repository, submit issues, or create pull requests with improvements or additional augmentation techniques.

## License

This project is intended for educational purposes. The dataset may have its own usage restrictions.


# Note

This notebook is designed for educational and research purposes. Real-world applications may require additional considerations and validation.

