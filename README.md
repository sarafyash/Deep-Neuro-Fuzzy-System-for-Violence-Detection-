# Deep-Neuro-Fuzzy-System-for-Violence-Detection
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

## Data Pre-Processing:
In the data pre-processing phase, videos are processed to extract individual frames using the cv2 library from OpenCV. This step involves breaking down each video into its constituent frames, which are then resized to a standardized dimension of 64x64 pixels. A sequence length of 16 frames per video is established to ensure effective temporal information processing. Videos with fewer than 16 frames are filtered out to maintain dataset consistency. Two classes, namely "Non-Violence" and "Violence," are defined as target categories for the model. The dataset is organized into arrays containing features (frames), corresponding labels (one-hot-encoded vectors representing class indices), and paths to video files. Finally, the dataset is split into 80% training and 20% testing sets to facilitate model evaluation and validation.

## MobileNetV2 as Feature Extractor:
MobileNetV2 is employed as a feature extractor in the model architecture. This lightweight convolutional neural network (CNN) architecture is chosen for its efficiency in extracting spatial features from video frames. Utilizing pre-trained weights, MobileNetV2 is able to capture meaningful spatial information while maintaining computational efficiency. In the context of this model, only the convolutional layers of MobileNetV2 are retained, and the fully connected layers are discarded. This adaptation ensures that the network focuses solely on extracting relevant features from the input video frames without unnecessary processing.

  ### Fine-Tuning MobileNetV2:
  To tailor MobileNetV2 specifically for the task of violence detection, a process known as fine-tuning is applied. Fine-tuning allows the model to adapt its pre-trained weights to better suit the characteristics   of the dataset and the specific classification task. In this case, the last 40 layers of MobileNetV2 are fine-tuned during training, while the earlier layers are frozen. This selective fine-tuning approach     
  helps prevent overfitting by adjusting only the most relevant parameters, thereby enhancing the model's ability to learn discriminative features related to violence detection.

## Bidirectional LSTM (BiLSTM) for Modeling:
After extracting spatial features using MobileNetV2, the sequence of feature maps is passed through a Bidirectional Long Short-Term Memory (BiLSTM) layer. BiLSTMs are a type of recurrent neural network (RNN) that excel in capturing temporal dependencies in sequential data. By processing information in both forward and backward directions, BiLSTMs effectively capture context from past and future frames within each video sequence. This capability is crucial for understanding temporal patterns and dynamics, thereby enhancing the model's ability to discriminate between violent and non-violent behavior in videos.

## Fully Connected Layers for Neuro-Fuzzy Classification:
The model incorporates fully connected dense layers to further process the high-level spatiotemporal features extracted by MobileNetV2 and BiLSTM. These layers are pivotal in transforming the learned features into actionable predictions regarding the presence of violence in video sequences. Rectified Linear Unit (ReLU) activation functions introduce non-linearity, allowing the model to capture complex patterns inherent in video data. Dropout regularization techniques are applied to prevent overfitting and enhance the model's generalizability. The final layer employs a sigmoid activation function to produce class probabilities, indicating the likelihood of violence in the input video sequence. This comprehensive approach integrates feature extraction, sequence learning, and classification to achieve accurate violence detection in video data.

## Working 

To see the code in action, please refer to the following GitHub repository:
[GitHub - Deep Neuro Fuzzy System for Violence Detection](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/blob/main/Capstone_Project.ipynb).

## Results and Discussion

Our study investigates automated violence detection in videos, crucial for safety and content moderation. Using diverse datasets including movies, sports videos, and online content, our model achieved an initial accuracy of 97.16% without additional techniques. Integration of SVM, Naïve Bayes, Random Forest, K-NN, and Gradient Boosting improved accuracy, with Naïve Bayes reaching 97.87%. This validates combining fuzzy logic and neural networks for enhanced classification.

Our model evaluates each video frame thoroughly, providing binary violence predictions with 99% confidence. It supports automated safety measures and efficient content moderation in online platforms. Visual aids including accuracy graph, confusion matrix , and classification report detail system performance.

  ![accuracy (2)](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/9786af72-f39f-4f97-93b6-87f7423e6a5b)
  
  ![Confusion Matrix (1)](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/8e5d3f29-c1ee-4b59-8d5e-ca28988aff89)
  
  ![precision](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/aec30e8b-99b3-49b1-b603-9c010fa1ce9a)

Performance metrics for "Not Violence" and "Violence" classes show high precision, recall, and F1-score, averaging 97% accuracy across 423 instances.

## Neuro-Fuzzy Classification System

We began by evaluating our model's performance without additional classification techniques, achieving an initial accuracy of 97.16%. Integrating SVM, Naïve Bayes, Random Forest, K-NN, and Gradient Boosting classifiers into our neuro-fuzzy system showed varied accuracy improvements. Naïve Bayes achieved the highest accuracy at 97.87%, while SVM, Gradient Boosting, and Random Forest also improved significantly. K-NN, however, exhibited comparatively lower accuracy rates. This highlights the importance of selecting appropriate classification techniques to maximize violence detection system effectiveness.

  ![Neurofuzzy](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/00404cba-9253-4c65-9deb-c4b0ceaa0002)

## Predictions

The violence detection system achieves precise identification of violent videos with a confidence level of 99.629%, shown in below. Similarly, it accurately distinguishes non-violent content, classifying videos with a confidence level of 99.177%.

This version maintains clarity while emphasizing the high confidence levels in both violence and non-violence classifications.

![Violence_Prediction (1)](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/3826100f-e027-4d54-bdac-1d795a886660)
![NonViolence_Prediction (1)](https://github.com/sarafyash/Deep-Neuro-Fuzzy-System-for-Violence-Detection-/assets/88444976/5f007c99-3fcb-4329-a89e-cefb53b28472)


## Conclusions

In conclusion, our hybrid CNN-RNN architecture effectively identifies violent video content by leveraging MobileNetV2 for spatial features and BiLSTM for temporal dynamics. Incorporating neuro-fuzzy classification techniques further enhances accuracy, demonstrating efficacy across diverse video datasets. Significant improvements in accuracy, notably from 96.37% to 97.87%, underscore the model's robustness. While facing limitations in frame size and computational complexity, future research can focus on adaptive model architectures and optimization strategies to enhance practical viability and scalability in real-world settings. This study represents a significant advancement in violence detection, promising improved security measures across applications from surveillance to social media platforms.
