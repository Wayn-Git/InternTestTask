# 💸 Fraud Detection System

A comprehensive machine learning solution for detecting fraudulent financial transactions using Random Forest classification with advanced feature engineering and data balancing techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Technical Approach](#technical-approach)
- [Requirements](#requirements)
- [Results](#results)

## Overview

This project implements a fraud detection system that identifies fraudulent financial transactions from a highly imbalanced dataset. The solution addresses common challenges in fraud detection including:

- **Class imbalance** (fraud cases are extremely rare)
- **Feature leakage** prevention
- **Temporal patterns** in fraudulent activity
- **Transaction type analysis**
- **Amount-based fraud patterns**

## Dataset

The system uses a financial transaction dataset (`Fraud.csv`) containing the following features:

## Dataset

The system uses a financial transaction dataset (`Fraud.csv`) containing the following features:

- **step** - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation). 
- **type** - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
- **amount** - amount of the transaction in local currency.
- **nameOrig** - customer who started the transaction
- **oldbalanceOrg** - initial balance before the transaction
- **newbalanceOrig** - new balance after the transaction
- **nameDest** - customer who is the recipient of the transaction
- **oldbalanceDest** - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
- **newbalanceDest** - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
- **isFraud** - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
- **isFlaggedFraud** - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

## Features

### 🔍 Data Analysis
- Comprehensive exploratory data analysis (EDA)
- Missing value analysis using missingno
- Class distribution visualization
- Transaction type fraud pattern analysis
- Temporal fraud pattern analysis

### 🔧 Advanced Feature Engineering
- **Merchant identification flags**
- **Financial ratio calculations**
- **Temporal features** (hour, day, night/weekend indicators)
- **Transaction type encoding**
- **Amount-based pattern detection**
- **Risk indicator flags**

### ⚖️ Data Balancing
- SMOTE (Synthetic Minority Over-sampling Technique) implementation
- Addresses extreme class imbalance (fraud cases < 1%)
- Maintains data diversity to prevent overfitting

### 🤖 Machine Learning Model
- Random Forest Classifier with class balancing
- Feature importance analysis
- Robust evaluation metrics (AUC-ROC, AUC-PR, Classification Report)

## Data Analysis

### Key Insights Discovered:

1. **Extreme Class Imbalance**: Fraud cases represent less than 1% of all transactions
2. **Transaction Type Patterns**: 
   - CASH_OUT and TRANSFER transactions show highest fraud rates
   - PAYMENT, DEBIT, and CASH_IN transactions rarely involve fraud
3. **Merchant Analysis**: Transactions to merchant accounts (starting with 'M') show different patterns
4. **Temporal Patterns**: No significant daily trends in fraud occurrence
5. **Amount Patterns**: Large, round amounts are more likely to be fraudulent

## Feature Engineering

### Safe Features (No Data Leakage):
```python
safe_features = [
    'step',                    # Transaction timestamp
    'amount',                  # Transaction amount
    'oldbalanceOrg',          # Origin balance before transaction
    'oldbalanceDest',         # Destination balance before transaction
    'type_encoded',           # Encoded transaction type
    'is_merchant_dest',       # Destination is merchant flag
    'is_merchant_orig',       # Origin is merchant flag
    'amount_to_orig_balance', # Amount to origin balance ratio
    'amount_to_dest_balance', # Amount to destination balance ratio
    'hour_of_day',           # Hour of transaction
    'day_of_simulation',     # Day of transaction
    'is_night',              # Night time transaction flag
    'is_weekend',            # Weekend transaction flag
    'is_large_amount',       # Large amount flag (≥200,000)
    'is_round_amount'        # Round amount flag (multiples of 50,000)
]
```

### Avoided Features (Potential Leakage):
- `newbalanceOrig`: Balance after transaction (outcome dependent)
- `newbalanceDest`: Destination balance after transaction (outcome dependent)

## Model Performance

The model is evaluated using multiple metrics appropriate for imbalanced classification:

- **AUC-ROC Score**: Area under the ROC curve
- **AUC-PR Score**: Area under the Precision-Recall curve
- **Classification Report**: Precision, Recall, F1-Score for both classes
- **Confusion Matrix**: Detailed breakdown of predictions

### Why These Metrics?
Standard accuracy is misleading with imbalanced data. Our focus on Precision, Recall, and F1-Score provides better insight into fraud detection performance.

## Key Findings

1. **Transaction Types Matter**: CASH_OUT and TRANSFER are primary fraud vectors
2. **Timing Patterns**: While no strong daily patterns exist, hour-of-day features still provide value
3. **Amount Significance**: Large, round amounts are strong fraud indicators
4. **Balance Ratios**: Transaction amount relative to account balances is predictive
5. **Merchant Transactions**: Different fraud patterns for merchant vs. individual accounts

## Technical Approach

### Data Preprocessing
1. **Missing Value Analysis**: Visualized using missingno matrix
2. **Feature Creation**: engineered features from raw data
3. **Encoding**: Label encoding for categorical variables
4. **Scaling**: Handled implicitly by Random Forest

### Model Selection
- **Random Forest Classifier**: Chosen for robustness to outliers and feature importance
- **Class Balancing**: Both SMOTE oversampling and class_weight parameter
- **Cross-validation**: Stratified splitting to maintain class distribution

### Evaluation Strategy
- **Stratified Train-Test Split**: Maintains class balance across splits
- **Multiple Metrics**: Comprehensive evaluation beyond simple accuracy
- **Feature Importance**: Analysis of most predictive features

## Requirements

```
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
missingno>=0.4.2
```

---

**Note**: This system is designed for educational and research purposes. For production fraud detection systems, additional considerations including regulatory compliance, real-time processing, and continuous model monitoring are essential.
