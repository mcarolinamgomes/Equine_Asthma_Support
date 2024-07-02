# Evaluating Asthma in Equines with Video Recordings

#### Repository Overview
This repository contains the code and datasets associated with the paper "Evaluating Asthma in Equines with Video Recordings" by Carolina Gomes, Paula Tilley, and Luísa Coheur. The research focuses on developing a non-invasive, video-based diagnostic model to detect asthma symptoms in horses, primarily utilizing computer vision and machine learning techniques.

#### Contents
- **Datasets**:
  - **ASTHMA dataset**: Contains image subtraction and segmentation frames of recordings of 23 horses (10 asthmatic and 13 healthy).

- **Code**:
  - **Feature Engineering Method**: Utilizes classic machine learning classifiers trained on features extracted from segmented images of nostrils and abdomen.
  - **Image Subtraction Method**: Applies image subtraction techniques on consecutive frames, enabling motion detection without complex 3D methods. Various image classification models (VGG, ResNet, MobileNet, EfficientNet) are evaluated.

- **Modeling and Evaluation**:
  - Implementation details of classifiers (SVM, Decision Tree, Random Forest, XGBoost) and their performance metrics.
  - Cross-validation strategy and detailed results showcasing model performance.

#### Key Features
- **Non-invasive Diagnostic Model**: A user-friendly and cost-effective approach suitable for use in natural environments, reducing the need for clinical visits.
- **Machine Learning Techniques**: Application of both classic and deep learning models for robust asthma detection.
- **Comprehensive Evaluation**: Comparative analysis of different methods and classifiers to identify the best-performing models.

#### How to Use
1. **Data Preparation**: Follow instructions to preprocess and segment video frames using the provided datasets.
2. **Model Training**: Train the machine learning models using the provided scripts.
3. **Evaluation**: Evaluate the models using the test datasets and compare their performance metrics.

#### Conclusion and Future Work
The repository provides a foundation for developing advanced models for equine asthma detection. Future work involves expanding the dataset and refining the models to enable accurate asthma staging in non-hospital settings, improving the quality of life for performance animals.

#### References
Refer to the original paper for detailed methodologies, experimental setup, and comprehensive analysis:
- [Evaluating Asthma in Equines with Video Recordings](link-to-paper)

#### Acknowledgments
This work was supported by national funds through FCT, Fundação para a Ciência e a Tecnologia, under project UIDB/50021/2020.

---

Feel free to explore the repository, contribute, or use the provided tools to advance your research in equine health diagnostics!
