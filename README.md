# Credit Card Default Prediction in Taiwan

## Overview

This project aims to predict the likelihood of credit card defaults among clients in Taiwan using a dataset provided by the UCI Machine Learning Repository. By leveraging client information and historical transaction data, the goal is to create a reliable predictive model that can help banks identify individuals likely to default on their credit cards. This can enable proactive measures to mitigate potential defaults and support customers in managing their financial obligations.

## Motivation

In Taiwan, aggressive issuance of credit cards led to widespread debt accumulation, with severe societal repercussions. Effective crisis management and robust risk prediction are critical to stabilizing the financial system. This project aims to develop a predictive model to forecast individual credit risk, thereby reducing potential damage and uncertainty in the financial sector.

## Dataset

The dataset contains information on 30,000 clients, with features such as:

- **LIMIT_BAL**: Amount of the given credit (NT dollar)
- **SEX**: Gender (1 = male; 2 = female)
- **EDUCATION**: Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others)
- **MARRIAGE**: Marital status (1 = married; 2 = single; 3 = others)
- **AGE**: Age (year)
- **PAY_0 to PAY_6**: History of past payment (from April to September 2005)
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement (from April to September 2005)
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payment (from April to September 2005)
- **Default**: Binary variable indicating default payment (1 = default; 0 = no default)

## Installation

To run the analysis, you need the following libraries installed:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Exploratory Data Analysis (EDA)

- **Correlation Analysis**: Identified highly correlated features to avoid multicollinearity.
- **Distribution Analysis**: Examined the distribution of default status, gender, education, and age.

## Principal Component Analysis (PCA)

PCA is used to reduce dimensionality, retaining 99% of the variance with 19 components instead of all 25. This helps simplify the model and reduces computational complexity.

## Model Building and Evaluation

Multiple models were built and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost**
5. **AdaBoost**

Hyperparameters for each model were optimized using GridSearchCV with cross-validation. Models were evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC score.

## Results

- **Random Forest**: Best performance with an accuracy of 81.55%.
- **XGBoost**: Accuracy of 81.33%.
- **AdaBoost**: Accuracy of 80.80%.
- **Logistic Regression and Decision Tree**: Performed well but with lower accuracy compared to ensemble methods.

## Conclusion

The Random Forest model performed the best, making it the preferred choice for predicting credit card defaults in this dataset. However, XGBoost offers a good trade-off between accuracy and computational efficiency.

## Future Work

- Improve model performance on the minority class (default cases) by exploring advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique).
- Experiment with other ensemble methods and deep learning approaches for further accuracy improvements.
- Implement the model in a real-world financial system to provide proactive solutions for potential defaulters.

## Acknowledgments

This project uses data from the UCI Machine Learning Repository: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## License

This project is licensed under the MIT License.
