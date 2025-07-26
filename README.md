# 📈 Stock Price Trend Prediction using LSTM

This project predicts future Tata Motors stock prices using historical data and an LSTM (Long Short-Term Memory) deep learning model.
📌 Project Overview
Domain: Time Series Forecasting
Algorithm: LSTM Neural Network
Dataset: Tata Motors stock price data
Download from Kaggle: NIFTY‑50 Stock Market Data by Rohan Rao
Includes historical CSV files for Tata Motors

- *Dataset:* Tata Motors stock data (Kaggle)
 - 🔗 [Kaggle Dataset – NIFTY50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)

## 📌 Overview
- Domain: Time Series Forecasting  
- Algorithm: LSTM Neural Network  
- Dataset: Tata Motors stock data (Kaggle)  
- Tools: Python, Pandas, Keras, TensorFlow, Matplotlib  
- Platform: Google Colab / Jupyter Notebook  

## 🔄 Workflow
1. Upload and load tatamotors.csv  
2. Normalize the closing prices  
3. Create time series sequences (60 timesteps)  
4. Build and train the LSTM model  
5. Predict and visualize future prices

🧠 Top 50 Interview Questions with Answers:
1.What is the difference between Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL)?
- AI is general intelligence; ML uses data to learn patterns; DL uses neural networks.
2.What are the different types of Machine Learning?
- Supervised, Unsupervised, Reinforcement Learning.
3.What is the difference between Supervised and Unsupervised Learning?
- Supervised uses labeled data; unsupervised uses unlabeled data.
4.What is Overfitting and Underfitting in Machine Learning?
- Overfitting memorizes data; underfitting can't learn patterns.
5.What are common methods to evaluate a model’s performance?
- Accuracy, Precision, Recall, F1-score, ROC-AUC.
6.What is the Bias-Variance Trade-off?
- Balance between underfitting and overfitting.
7.What is Cross-Validation in Machine Learning?
- Technique to check model performance on multiple subsets.
8.What is the difference between Precision, Recall, F1 Score, and Accuracy?
- Precision: TP / (TP + FP), Recall: TP / (TP + FN), F1 is their harmonic mean.
9.What is the difference between Classification and Regression?
- Classification predicts categories, regression predicts numbers.
10.What are real-world applications of Machine Learning?
- Chatbots, spam filters, fraud detection, recommendation systems.
11.How do you handle missing data in a dataset?
- Drop rows, fill with mean/median, or use KNN imputation.
12.What is the difference between Normalization and Standardization?
- Normalization scales data to [0,1]; standardization centers it with mean = 0 and std = 1.
13.What is One-hot Encoding and why is it used?
- It converts categorical data into binary vectors for ML models.
14.How do you handle categorical data in machine learning?
-Using one-hot encoding, label encoding, or embeddings.
15.What is Feature Selection and why is it important?
-It reduces dimensionality, improves performance, and avoids overfitting.
16.What is Principal Component Analysis (PCA)?
- PCA is a technique to reduce feature dimensions by combining correlated variables.
17.What is the Curse of Dimensionality?
- Too many features can cause model overfitting and poor generalization.
18.What is Feature Scaling and when is it necessary?
- It ensures features contribute equally during training; used in distance-based models.
19.How do you handle imbalanced datasets?
- Use resampling methods like SMOTE, undersampling, or set class weights.
20.What is Exploratory Data Analysis (EDA)?
- A process to analyze datasets using statistics and visualizations before modeling.
21.What is a Decision Tree and how does it work?
- It splits data into branches based on feature values for classification or regression.
22.What is the difference between Bagging and Boosting?
- Bagging reduces variance by averaging; boosting reduces bias by sequential correction.
23.How does the K-Nearest Neighbors (KNN) algorithm work?
- It predicts based on the majority label of the 'k' nearest data points.
24.What is Support Vector Machine (SVM)?
- It finds the best hyperplane to separate different classes in the dataset.
25.What is Naive Bayes and how is it used?
- A probabilistic classifier based on Bayes’ theorem assuming feature independence.
26.What is the difference between Random Forest and XGBoost?
- Random Forest is bagging-based; XGBoost is a gradient boosting technique.
27.What is the difference between Gradient Descent and Stochastic Gradient Descent?
- GD uses the whole dataset; SGD uses one sample at a time for updates.
28.What is Logistic Regression used for?
- For binary classification problems.
29.What are Ensemble Models in Machine Learning?
- They combine multiple models to improve prediction accuracy.
30.What is the difference between L1 and L2 Regularization?
- L1 adds absolute value penalty (sparse model); L2 adds squared weight penalty.
31.What is a Perceptron in Neural Networks?
- A basic computational unit that performs binary classification.
32.What are Activation Functions and why are they used?
- Functions like ReLU, Sigmoid, and Tanh add non-linearity to neural networks.
33.What is the role of Epoch, Batch Size, and Learning Rate in training?
- Epoch = 1 full dataset pass, Batch = subset for training, LR = step size for updates.
34.What is the Vanishing Gradient Problem?
- A deep network issue where gradients become too small to update weights.
35.What is the difference between CNN and RNN?
- CNNs work with images; RNNs handle sequences like time-series or text.
36.What is an LSTM network?
- A special type of RNN that remembers long-term dependencies using memory cells.
37.What is a Convolutional Layer in CNNs?
- It applies filters to extract spatial features from input data.
38.What is Transfer Learning?
- Reusing a pre-trained model on a new but similar task.
39.What is Dropout in neural networks?
- Randomly turns off neurons during training to prevent overfitting.
40.How can you prevent overfitting in ML models?
- Use regularization, dropout, early stopping, and more data.
41.What is Tokenization in NLP?
- Breaking down text into smaller parts like words or subwords.
42.What are Word2Vec and GloVe embeddings?
- Techniques to convert words into dense vectors that capture meaning.
43.What is the difference between Stemming and Lemmatization?
- Stemming removes suffixes crudely; lemmatization uses vocab and grammar.
44.What is TF-IDF and where is it used?
- It measures the importance of words in documents.
45.What are Transformers in deep learning?
- Models that use attention mechanisms (e.g., BERT, GPT).
46.How can you save and load a machine learning model?
- Use pickle, joblib, or model.save() for deep learning models.
47.What is Model Drift?
- When a deployed model’s performance decreases due to new patterns in data.
48.How do you deploy a machine learning model?
- Use APIs like Flask, FastAPI or platforms like Streamlit, Heroku, AWS.
49.What are common deployment tools for ML?
- Docker, Streamlit, Flask, AWS, Azure.
50.What is CI/CD in MLOps?
- Continuous Integration/Deployment automates testing and delivery of ML pipelines.
