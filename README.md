Title:Customer Segmentation Using K-means Clustering
 Customer Segmentation Analysis
 Project Overview

A comprehensive customer segmentation analysis system that clusters mall customers into distinct groups based on their purchasing behavior, demographics, and spending patterns. This web application provides interactive clustering capabilities with real-time visualization.



 Project Objective

To develop an interactive web application that enables businesses to:
Segment customers into meaningful groups using machine learning
Visualize customer clusters with interactive plots
Understand customer behavior patterns for targeted marketing

Make data-driven decisions for customer relationship management

 Dataset Details
Source
Dataset: Mall Customer Segmentation Data
Samples: 500 customers
Features: 5 attributes

Feature Description
Feature	Description	Type
CustomerID	Unique customer identifier	Numerical
Gender	Customer gender (Male/Female)	Categorical
Age	Customer age in years	Numerical
Annual Income (k$)	Annual income in thousands of dollars	Numerical
Spending Score (1–100)	Spending behavior score	Numerical
Key Statistics

Average Age: 38.85 years

Average Annual Income: $60.56k

Average Spending Score: 50.20

Gender Distribution: 56% Female, 44% Male

Algorithms/Models Used
1. K-Means Clustering
Purpose: Unsupervised learning for customer segmentation
Key Parameters:
n_clusters: 2–10
init: k-means++
random_state: 42

2. Cluster Validation Methods
Elbow Method (WCSS)
Silhouette Score for cluster separation

3. Data Preprocessing
StandardScaler for numerical features
One-hot encoding for categorical features
Supports uni-, bi-, and multivariate clustering

 Clustering Approaches
1. Univariate Clustering
Feature: Annual Income
Typical Clusters: 3
Use Case: Income-based segmentation

2. Bivariate Clustering (Recommended)
Features: Annual Income + Spending Score
Optimal Clusters: 5
Use Case: General customer profiling

3. Multivariate Clustering
Features: Age, Income, Spending Score, Gender
Clusters: User-defined
Use Case: High-accuracy segmentation

 Results & Performance
Optimal Configuration
Recommended k: 5
Silhouette Score: ~0.55–0.60
WCSS: Lowest at k=5

Customer Segments Identified
Cluster	Profile	Income	Spending	Size
0	High Income, Low Spending	High	Low	~20%
1	Avg Income, Avg Spending	Medium	Medium	~20%
2	High Income, High Spending	High	High	~20%
3	Low Income, High Spending	Low	High	~20%
4	Low Income, Low Spending	Low	Low	~20%

Visualizations Produced
Age, Income, Spending Score distributions
Gender-based income comparison
Correlation heatmap
Cluster scatter plots
Silhouette score visualization

 Installation & Setup
Prerequisites
Python 3.8+
pip package manager
Steps

 Clone repository
git clone <repository-url>
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt
# Run application
python app.py
Access
Open browser → http://localhost:5000

Required Libraries
Flask==2.3.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
numpy==1.24.3

 Usage Guide
Upload CSV dataset
Explore data
Configure clustering (2–10 clusters)
View results
Download segmented data

 Project Structure
customer-segmentation/
│
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── analysis.html
│   └── results.html
├── static/
│   └── style.css
└── README.md

 Key Findings
Business Insights
Five distinct customer groups exist
Spending correlates strongly with income
High-income, high-spending customers are most valuable
High-income, low-spending segment is a marketing opportunity

Technical Achievements
Silhouette score > 0.55
Scalable to 10,000+ users
Intuitive UI
Multiple clustering modes

Conclusion
The system:
Applies ML clustering effectively
Provides real-time visualization
Offers actionable business insights
Supports flexible clustering approaches
Enables targeted customer segmentation

 Future Scope
Enhanced Features
Real-time clustering
DBSCAN & hierarchical models
Automatic cluster naming
CRM integration
Technical Upgrades
Database storage
Authentication
REST API
Docker + cloud deployment

Advanced Analytics
Time-series spending
Recommendation engine
Churn prediction
A/B test framework

 References
Hemashree Kilari, Sailesh Edara, Guna Ratna Sai Yarra, Dileep Varma Gadhiraju, “Customer Segmentation using K-Means Clustering,” IJERT, Vol. 11, Issue 03, March 2022
Anjana K. Mahanta, Amar J. Singh, Th. Shanta Kumar, “Customer Segmentation: Using a Comparative Case of Clustering Algorithms,” Data Mining & Knowledge Engineering.
Yulan Zheng, “Customer Segmentation Research in Marketing through Clustering Algorithm Analysis,” Journal of Intelligent & Fuzzy Systems, Vol. (2023). 
Desi Adrianti Awaliyah, Budi Prasetiyo, Rini Muzayanah, Apri Dwi Lestari, “Optimizing Customer Segmentation in Online Retail Transactions through the Implementation of the K-Means Clustering Algorithm,” Scientific Journal of Informatics.

Documentation
Scikit-learn Clustering
Flask Framework
Matplotlib Guide
Datasets
Kaggle Mall Customers

UCI Repository
 Contributors
Purva Thombare
Tanuja Birajdar
Shruti Malkar

