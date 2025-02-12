# Sentiment Analysis Assignment  

## 📌 Overview  
This project performs sentiment analysis on IMDB movie reviews using both traditional machine learning and deep learning techniques. We implemented and compared **Logistic Regression** and **LSTM** models to classify reviews as positive or negative.  

## 📚 Dataset  
- **Source**: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Size**: 50,000 reviews (25,000 positive, 25,000 negative)  
- **Task**: Binary classification (Positive vs. Negative sentiment)  

## 🛠 Preprocessing Steps  
- Removed special characters, HTML tags, and stopwords  
- Tokenized and lemmatized text  
- Applied **TF-IDF** for feature extraction (Logistic Regression)  
- Used **Word2Vec embeddings** for LSTM model  

## 🏰 Model Implementation  
We trained and evaluated two models:  
### 1️⃣ **Logistic Regression**  
- Used **TF-IDF vectorization** for feature extraction  
- Achieved **86% accuracy** with regularization tuning  

### 2️⃣ **LSTM (Long Short-Term Memory)**  
- Used **Word2Vec embeddings** for input representation  
- Achieved **90% accuracy** with optimized hyperparameters  

## 🔍 Model Performance  
| Model | Accuracy | Precision | Recall | F1-Score |  
|--------|---------|----------|--------|---------|  
| Logistic Regression | 86% | 87% | 85% | 86% |  
| LSTM | 90% | 91% | 89% | 90% |  

## 📊 Results and Key Findings  
- **LSTM outperformed Logistic Regression** by capturing sequential dependencies in text.  
- **Hyperparameter tuning** significantly improved LSTM performance.  
- **TF-IDF worked well for Logistic Regression**, but deep learning performed better overall.  

## 🚀 How to Run the Project  
### 1️⃣ Install Dependencies  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk wordcloud
```
### 2️⃣ Clone the Repository  
```bash
git clone https://github.com/NdanyuzweP/Sentiment-Analysis-Assignment.git
cd Sentiment-Analysis-Assignment
```
### 3️⃣ Run the Notebook  
- Open `Sentiment_Analysis.ipynb` in **Jupyter Notebook** or **Google Colab**  
- Follow the steps in the notebook to preprocess data, train models, and evaluate performance  

## 📌 Team Contributions  
| Team Member | Role |  
|-------------|------------------------------|  
| **Prince Ndanyuzwe** | EDA & Logistic Regression Model |  
| **Cynthia Nekesa** | Data Preprocessing & LSTM Model |  
| **Smart Israel** | Model Evaluation & Report Writing |  

## 💎 References  
- [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- [How to Evaluate Sentiment Analysis Models](https://www.linkedin.com/advice/1/how-can-you-evaluate-sentiment-analysis-model-ygfec)  
- [Colab Notebook](https://colab.research.google.com/drive/1TieizSNC46iVaucxP_I7evXkNBDm9UCY#scrollTo=gGOVfub97sQj)  
