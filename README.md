# Malaria Detection using Deep Learning Model
This repository contains a deep learning model based on Convolutional Neural Networks (CNN) for the detection of malaria-infected and uninfected cell images. The model has been trained on a dataset obtained from Kaggle, and due to limited computational resources, only 100 samples of image data were processed.

# Dataset
The dataset used for training the model consists of cell images that are labeled as either infected or uninfected. The dataset was obtained from Kaggle and is a subset of the original dataset.

# Model Architecture
The deep learning model is built using the TensorFlow and scikit-learn libraries. It consists of 4 Convolutional blocks and 4 Max-Pooling blocks, which help in extracting important features from the input images. The model architecture also includes a classification layer that utilizes the softmax function for classifying the cell images as infected or uninfected.

# Results
After training the model, the achieved Final Validation Accuracy is 0.7799999713897705. It is important to note that this accuracy is based on the limited dataset of 100 samples. To further enhance the model's performance, it is recommended to train it on a larger and more diverse dataset. Additionally, hyperparameter tuning can also be performed to optimize the model's configuration.

# Visualizing Predicted Values and Ground Truth
To gain insights into the model's predictions, the predicted values are visualized alongside the ground truth labels. This visualization provides an overview of how well the model performs in classifying the cell images as infected or uninfected.

# Usage
To use the model, follow these steps:

1. Clone this repository: git clone https://github.com/yemioyeleke/Malaria_Detection_CNN.git

2. Install the required dependencies: pip install tensorflow scikit-learn matplotlib

3. Prepare your dataset or use the provided dataset in the appropriate format.

4. Adjust the model architecture, hyperparameters, and data preprocessing if needed.

5. Train the model using the provided code.

6. Evaluate the model's performance using validation data.

7. Visualize the predicted values and ground truth using the provided code.

8. Iterate on the model and experiment with different configurations to improve performance.

# Conclusion
This project demonstrates the development of a deep learning model for malaria detection based on a CNN architecture. Although the model achieved a reasonable accuracy with limited data, it has the potential for further improvement by utilizing a larger dataset and performing hyperparameter tuning. The visualizations of predicted values and ground truth provide insights into the model's performance and can aid in further analysis and refinement.

Please note that this implementation serves as a starting point and can be expanded upon to create a more robust malaria detection system.

For any inquiries or suggestions, please contact: masteroyeleke@gmail.com
