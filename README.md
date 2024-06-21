# Heart-Sound-Classification
# Project Description
Heart Sound Classification is a deep learning project aimed at classifying heart sounds to aid in the diagnosis of cardiovascular conditions. This project employs transfer learning using the AlexNet architecture and various signal processing techniques to classify heart sounds as normal or abnormal. It leverages MATLAB for data preprocessing, feature extraction, and model training.
# Dataset: PhysioNet Challenge 2016
This project uses the heart sound recordings from the PhysioNet/Computing in Cardiology Challenge 2016 dataset. The dataset includes heart sound recordings from different sources and labels them as normal or abnormal.

Visit the PhysioNet Challenge 2016 page.

Register for a PhysioNet account if you do not already have one.

Agree to the data usage terms and download the dataset.

Extract the dataset to a local directory, ensuring the folder structure matches the expectations in the provided code.
# Features
Heart Sound Data Processing: Reads and processes heart sound recordings from multiple datasets.

Signal Filtering: Applies bandpass filtering to heart sound signals.

Feature Extraction: Uses Local Binary Pattern (LBP) features extracted from spectrograms of heart sound signals.

Deep Learning Model: Implements transfer learning with AlexNet to classify heart sounds.

Performance Evaluation: Calculates and displays performance metrics including accuracy, sensitivity, specificity, precision, and F1 score.
# Technologies Used
Programming Languages: MATLAB

Libraries and Toolboxes: Signal Processing Toolbox, Image Processing Toolbox, Deep Learning Toolbox

Deep Learning Framework: AlexNet for transfer learning

