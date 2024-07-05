# Deep-Neuro-Fuzzy-System-for-Violence-Detection-
### Violence detection using MobileNet-V2 with Bi-Directional LSTM and Neuro-Fuzzy Classification System.

This project introduces a deep neuro-fuzzy system for violence detection in video content, combining BiLSTM networks, MobileNet architecture, and fuzzy logic. Achieving 97.87% accuracy, it offers robust and efficient detection, adaptable for various applications such as surveillance and social media monitoring.

## Introduction

With the increasing demand for violence detection in video content, this paper introduces an innovative deep neuro-fuzzy system. This system, which merges Bidirectional Long Short-Term Memory (BiLSTM) networks, MobileNet architecture, and fuzzy logic, offers a robust and swift violence detection solution. By combining BiLSTM’s temporal sequence analysis with MobileNet’s efficient spatial feature extraction, our system accurately identifies violent behavior while minimizing computational resources. Trained on a diverse dataset, the model achieved an impressive accuracy of 96.37% in violence detection before integrating neuro-fuzzy classification techniques. Subsequently, with the incorporation of these techniques, a notable enhancement was observed, with the accuracy further increasing to 97.87%. This improvement underscores the efficacy and potential of our proposed approach in achieving robust and accurate violence detection in real-world scenarios.Moreover, this research explores transfer learning for domain-specific fine-tuning, offering promising adaptability across various contexts such as surveillance and social media. The practical applications of our system are evident, as it can be used to enhance security measures and monitor online content. Evaluation metrics validate the model’s effectiveness, underscoring its superiority over existing methods.

## System Architecture

Our violence detection model integrates cutting-edge techniques from computer vision and machine learning into a sophisticated architecture. It utilizes MobileNet for efficient image extraction, CNN layers for spatial feature extraction, and a BiLSTM layer for temporal modeling. This design enables effective capture of spatiotemporal patterns crucial for violence detection in video segments. After sequence learning, features are processed through a fully connected dense layer and classified using a sigmoid activation function. Additionally, neuro-fuzzy classification techniques enhance classification accuracy and robustness by integrating SVM, Naive Bayes, Random Forest, K-NN, and gradient boosting classifiers. This integrated approach provides a comprehensive framework for automated violence detection, demonstrating efficacy and reliability in real-world applications.

Figure below visually represents the components and data flow within the violence detection system, illustrating the integration of CNNs, MobileNet, BiLSTM networks, neuro-fuzzy classification, and other modules for comprehensive video analysis and classification.

<img width="2272" alt="System Architecture Final (1)" src="https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/f8678fd5-90a6-40fc-b452-b2d1c1a1df0e">

## Dataset Description

Our dataset addresses a key limitation in existing research by offering unique diversity. It includes videos from popular sources used for violence detection such as the hockey fight dataset, movies dataset (7), and videos from YouTube and Shutterstock. The dataset is categorized into "Non-Violence" and "Violence" directories. The "Non-Violence" directory contains 1136 videos showing activities like eating, sports, and singing without violent content. The "Violence" directory includes approximately 1000 videos depicting severe violence in various scenarios, with video durations typically ranging from 2 to 7 seconds. This curated collection enhances the dataset's efficacy for developing violence detection algorithms.
