**Haul-Truck Collision Risk Classification**

ECE 607 – Final Project

**Project Description**

This project uses machine learning to classify simulated mine haul-truck encounters into three categories: Safe, Warning, and Critical.

The data was generated using the MineHaulSim simulator. The goal is to compare Logistic Regression and Gaussian Naive Bayes for classifying truck encounters.

Three models were tested:

1. Standard Logistic Regression
2. Balanced Logistic Regression
3. Gaussian Naive Bayes

The dataset was split into 80% training and 20% testing.

**How to Run**

Install the required Python packages:

  pip install -r requirements.txt

Run the dataset preparation script:

  python build_observed_risk_dataset.py

Run the machine learning comparison:

  python compare_observed_models.py

**Note:** File paths in the scripts may need to be updated to match the location of the CSV files on your computer.
