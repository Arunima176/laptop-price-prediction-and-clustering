**🖥️ Laptop Price Prediction and Clustering using Web Scraping & Machine Learning**

**📌 Project Overview**

This project focuses on extracting laptop specifications and prices from Flipkart using web scraping techniques, preprocessing the collected data, visualizing patterns, and applying both unsupervised and supervised machine learning models to cluster and predict laptop prices effectively.

**📂 Project Structure**

webscrapping_laptop_flipkart.ipynb – Scrapes laptop data from Flipkart using Selenium and BeautifulSoup.

laptops_dataset.csv – Raw dataset containing scraped laptop details.

scrapped_laptops.ipynb – Preprocessing, visualization, model training and evaluation notebook.

laptops-.db – SQLite database created using SQLAlchemy to store the cleaned dataset for future use.

final_svm_model.pkl – Final trained and tuned SVM model saved using joblib.

laptop_prediction.pptx – Presentation summarizing the entire project pipeline and results.

**📊 Data Description**

The dataset contains the following columns:

Title

Original Price

Discount Price

Rating

Number of Reviews

Offers

**🧹 Data Preprocessing**

Handled missing values

Standardized data types

Removed duplicates

Detected and removed outliers

Saved cleaned data into a database for future analysis using SQLAlchemy

**📈 Data Visualization**

Univariate and bivariate analysis

Price vs rating

Discounts vs original price

Histograms and scatter plots

**🤖 Unsupervised Learning**

K-Means Clustering used to group similar laptops

K=5 and K=2 clusters visu- Silhouette Score used to evaluate clustering quality

Domain knowledge suggested 2 primary clusters, indicating model captured sub-clusters

**🧠 Supervised Learning**

Several classification models were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

XGBoost Classifier

**🔧 Hyperparameter Tuning**

SVM was selected for final model

Used GridSearchCV to optimi C, gamma, and kernel

Initial accuracy: 96%

Final accuracy after tuning: 100%

Improved precision, recall, and F1-score

**✅ Conclusion**

Web scraping enabled the creation of a structured dataset from e-commerce data.

Data preprocessing and domain knowledge were crucial in building accurate models.

SVM, when fine-tuned, performed excellently with 100% accuracy.

Project demonstrates end-to-end data science process from data acquisition to deployment.

**🔗 Tools & Libraries**

Python, Pandas, NumPy, Matplotlib, Seaborn

Selenium, BeautifulSoup

Scikit-learn, XGBoost

SQLAlchemy, SQLite

GridSearchCV

