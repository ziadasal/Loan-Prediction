# Loan Prediction
![Loan Prediction](https://th.bing.com/th/id/OIP.rV3bJJcQISejb8vAGwXFdgHaDy?pid=ImgDet&w=1210&h=620&rs=1)
This repository contains a project for loan prediction, which aims to predict whether a loan applicant is likely to be approved or not based on various attributes.

## Project Explanation
### Data Collection
- The dataset is collected from [Kaggle](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset).
- The dataset which we get from kaggle consists of two CSV(Comma Separated Values) files.
  - One is Train Data (`train_u6lujuX_CVtuZ9i.csv`)
  - Another is Test Data (`test_Y3wMUE5_7gLdaTN.csv`)

**Loading the collected data**

- The CSV data is loaded with the help of [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method in pandas library.
```python
# TODO : To Load previous applicants loan application data
loan_train = pd.read_csv('../data/train_u6lujuX_CVtuZ9i.csv')
```
- The Training data consists of 614 applicant samples and 12 features.
## Dataset

The dataset used for this project is available in the file **loan_train**. It consists of the following attributes:

1. Loan ID: Unique identifier for each loan application
2. Gender: Gender of the applicant (Male or Female)
3. Married: Marital status of the applicant (Yes or No)
4. Dependents: Number of dependents the applicant has
5. Education: Education level of the applicant (Graduate or Not Graduate)
6. Self_Employed: Whether the applicant is self-employed or not (Yes or No)
7. ApplicantIncome: Income of the applicant
8. CoapplicantIncome: Income of the co-applicant (if any)
9. LoanAmount: Loan amount in thousands
10. Loan_Amount_Term: Term of the loan in months
11. Credit_History: Credit history of the applicant (1 for good credit history, 0 for bad credit history)
12. Property_Area: Area where the property of the applicant is located (Urban, Semiurban, or Rural)
13. Loan_Status: Whether the loan was approved or not (Y for approved, N for not approved)

## Goal

The goal of this project is to build a machine learning model that can predict whether a loan applicant will be approved or not based on the provided attributes. This can help banks and financial institutions automate the loan approval process and make more informed decisions.

## Implementation

The project includes the following files:

1. `loan_prediction.ipynb`: Jupyter Notebook containing the code implementation. It includes data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.
2. `loan_prediction.py`: Python script version of the Jupyter Notebook.
3. `loan_prediction.csv`: The dataset used for training and testing the model.
4. `README.md`: Documentation file providing an overview of the project.

## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- scikit-learn

Make sure to install these dependencies before running the code.

## Usage

To run the loan prediction model, follow these steps:

1. Clone this repository: `git clone https://github.com/ziadasal/Loan-Prediction.git`
2. Navigate to the cloned repository: `cd Loan-Prediction`
3. Open the Jupyter Notebook or execute the Python script to run the code.
4. The code will preprocess the data, train the machine learning model, and generate predictions for loan approval.

Feel free to modify the code or dataset as needed to explore different approaches or improve the model's performance.

## Conclusion

The loan prediction project demonstrates the application of machine learning in the banking sector. By training a model on historical loan data, it can provide valuable insights into loan approval decisions. However, keep in mind that the model's predictions are based on historical data and may not guarantee future loan approval outcomes. It is essential to regularly update and refine the model using the latest data for accurate predictions.
