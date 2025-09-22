# Deep Learning Portfolio

Welcome! 👋  
I am an aspiring Data Scientist showcasing my deep learning journey.  
This repository contains structured projects where I experiment with different architectures and datasets.  
I update this repository regularly to demonstrate consistency and growth in applied deep learning.

## 📂 Projects

1. **MNIST Baseline & Preprocessing**  
   - Built baseline dense NN on MNIST dataset  
   - Explored normalization, reshaping, and evaluation metrics  

2. **LeNet on MNIST**  
   - Implemented classic LeNet-5 CNN  
   - Achieved high accuracy on digit classification

3. **AlexNet**  
   - Implemented AlexNet architecture (convolutional layers, ReLU, normalization, large fully connected layers)  
   - Learned why ReLU massively speeds up convergence compared to older activations  
   - Applied data augmentation (random crops, flips, brightness) and dropout to reduce overfitting  
   - Observed the importance of GPU acceleration for training deep architectures on image datasets  
   - Experimented with training schedules, learning rate decay, and batch size to stabilize training  
   - Key intuition: AlexNet extracts hierarchical features — early layers learn edges and textures, deeper layers learn object parts and concepts

4. **VGG16**  
   - Implemented VGG16 with small (3×3) filters and deeper stacks of conv layers  
   - Studied trade-offs between depth and compute/memory cost  
   - Used VGG for transfer learning experiments (feature extraction + fine-tuning)

5. **Custom CNN Models**  
   - Built lightweight/custom CNNs tailored to smaller datasets  
   - Experimented with different block designs, pooling vs. strided conv, and batch normalization  
   - Focused on regularization (dropout, weight decay) to control overfitting

6. **Inception (GoogLeNet)**  
   - Implemented Inception-style modules with mixed filter sizes and 1×1 convolutions for dimensionality reduction  
   - Learned how multi-scale processing helps capture varied features without exploding compute cost

7. **Artificial Neural Network (Diabetes Dataset)**  
   - Built feedforward ANN for tabular (structured) data classification  
   - Performed feature scaling, missing-value handling, and basic feature engineering  
   - Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC

8. **Multiclass Classification with Deep Learning**  
   - Applied CNNs/ANNs to multiclass tasks using Softmax + categorical cross-entropy  
   - Explored label imbalance handling, class weights, and confusion-matrix analysis

---

## 🔬 What I learned (high-level)

- Proper **preprocessing and normalization** are essential before training.  
- **CNNs** are preferable for image tasks because they learn spatially-local features.  
- **Depth (VGG), width/multi-path (Inception), and regularization (dropout, batch norm)** are complementary ideas that improve representation and generalization.  
- **Data augmentation** and **learning rate scheduling** make a big difference on real-world performance.  
- **No single model** is best for all problems — architecture choice depends on dataset size, complexity, and compute constraints.

---

## 🚀 Tech Stack
- Python, TensorFlow / Keras, NumPy, Pandas, Matplotlib  
- Jupyter Notebooks / Google Colab  
- Local GPU (CUDA) for heavy training

---

## 📌 Next Steps
- Implement ResNet and experiment with skip-connections  
- Try transfer learning and fine-tuning on custom datasets  
- Explore GANs and generative modeling  
- Learn Transformers and sequence models (NLP / vision)  
- Deploy models as REST APIs and explore MLOps best practices

---

## 📎 How I work (short)
- Start with EDA & preprocessing → baseline model → progressively more complex architectures.  
- Track experiments (hyperparameters, train/val curves).  
- Use incremental improvements: augmentation → regularization → architecture changes → transfer learning.

---

## 📌 About Me
I am an aspiring Data Scientist passionate about deep learning.  
This repository documents my practical experiments, intuition, and the lessons I learned while building and refining neural networks.

---

> For details and code, open the notebooks in this repo (e.g., `01_MNIST_dataset.ipynb`, `02_LeNet-5.ipynb`, `03_AlexNet.ipynb`, `04_VGG16.ipynb`, etc.).  
> Feel free to raise an issue or PR — I welcome feedback and collaboration!
