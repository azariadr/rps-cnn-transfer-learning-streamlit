From Scratch to Streamlit: Building and Comparing CNN and Transfer Learning Models for a Rock-Paper-Scissors Game

# Overview
This project explores the fascinating world of deep learning by building and comparing two different approaches to classify hand gestures for a Rock-Paper-Scissors game: a Convolutional Neural Network (CNN) built from scratch and a Transfer Learning model using MobileNetV2. The journey culminates in deploying the best-performing model as an interactive web application using Streamlit.

This repository serves as a comprehensive demonstration of an end-to-end machine learning pipeline, from data preparation and model training to evaluation and interactive deployment.

# Key Features
- Data Preprocessing: Efficient loading, augmentation, and splitting of the Rock-Paper-Scissors image dataset.
- Custom CNN: Implementation and training of a CNN model from scratch for image classification.
- Transfer Learning: Utilization and fine-tuning of the pre-trained MobileNetV2 model for improved performance.
- Model Evaluation: Detailed analysis of both models using accuracy, loss plots, classification reports, and confusion matrices.
- Interactive Streamlit App: Deployment of the superior model as a fun, interactive Rock-Paper-Scissors game where users can upload hand gestures and play against an AI.

# Installation and Setup
To run this project locally, follow these steps:
- Clone the repository:
```
git clone https://github.com/YourUsername/rock-paper-scissors-dl-comparison.git
cd rock-paper-scissors-dl-comparison
```
(Remember to replace YourUsername with your actual GitHub username)

- Create a virtual environment (recommended):
```
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
- Install the required packages:
```
pip install -r requirements.txt
```
(You'll need to create a requirements.txt file based on your imports, e.g., tensorflow, streamlit, numpy, matplotlib, seaborn, scikit-learn)

- Download the dataset:
The dataset is automatically downloaded by the provided Python script. If running locally, ensure you have the rockpaperscissors.zip file extracted to a rockpaperscissors directory in your project root, or modify the data loading path in the scripts.
```
# This command is in the Jupyter/Colab notebook, but for local setup:
wget --no-check-certificate https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
unzip rockpaperscissors.zip -d .
```
- Ensure model files are present:
The trained model files (model_cnn_final.keras and model_tl_phase2_fine_tuned.keras) should be in the root directory of your project, or adjust the MODEL_PATH in the Streamlit app. These models are generated during the training phase described in the notebook.

# How to Run the Streamlit App
Once you have installed the dependencies and placed the model files, you can run the Streamlit application:
```
streamlit run app.py
```
This will open the application in your web browser, where you can interact with the Rock-Paper-Scissors game.

# Model Details and Results
This project compares two models:

## Custom Convolutional Neural Network (CNN)
- Architecture: A sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
- Performance: Achieved 99% accuracy on the test set.
- Key Takeaway: A strong baseline performance, demonstrating the effectiveness of CNNs for image classification.

## Transfer Learning with MobileNetV2
- Approach: Utilized a pre-trained MobileNetV2 model (frozen layers) as a feature extractor, followed by custom classification layers. This was then followed by a fine-tuning phase where the base model layers were unfrozen and trained with a very low learning rate.
- Performance: Achieved an outstanding 100% accuracy on the test set.
- Key Takeaway: Significantly outperformed the custom CNN, showcasing the power of leveraging pre-trained knowledge for faster convergence and superior generalization, especially with limited datasets.
