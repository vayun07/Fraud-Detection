# Fraud-Detection
Financial fraud is a major concern for businesses and consumers. Through Machine Learning and trained AI models, suspicious transactions can be deterred.

I.	Approach: Fraud Detection Model

a)	Building a test sample with Faker, a Python library, to generate synthetic data to create 1000 transactions which would be used as input to test the fraud detection model.

b)	Data Processed with Apache Spark and Spark DataFrame.

c)	Machine Learning with Spark MLlib is used to implement Random Forest Classifier, which is a supervised learning algorithm trained on labelled data to learn patterns that differentiate fraudulent activities.

II.	Constructing the Fraud Detection Model

a)	Generate test data with Faker.py file
b)	Building the Fraud Detection Pipeline with Fraud Detection.py 

Steps:

Initialize Spark Session -> Load Synthetic Data -> Preprocess Data (Index, Vector Translation) -> Fit and Transform Data -> Split Data -> Train Random Forest Classifier -> Evaluate Model Performance -> Display Data -> Stop Spark Session
