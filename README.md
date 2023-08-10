# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The purpose of this project is to implement clean code principles to a machine learning pipeline for identify credit card customers that are most likely to churn. As detailed below, the completed project consists of a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package has the flexibility of being run interactively through a Jupyter notebook or from the command-line interface (CLI).  Customer Churn is a condition in which consumers do not continue or leave the services offered by an industry. If not identified and handled, it can lead to loss of revenue. 

The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). 

The project proposes the following machine learning workflow:

1. __Import data__ from within the same repository.
2. __Exploratory data analysis:__  To examine the distribution of values and the correlations between features in data
3. __Feature engineering:__ To encode categorical features, standardize numerical features, and split data into training and test sets.
4. __Training models:__ To train two models using logistic regression and randome forest, perform hyper-parameter tuning, and evaluate the performance of each model using ROC curves and classification reports.

## Project Structure

The structure of this project directory tree is displayed as follows:

```
.
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Marital_Status.png
│   │   ├── Total_Trans_Amt.png
│   │   └── Heatmap.png
│   └── results
│       ├── Feature_Importances.png
│       ├── logistic_Results.png
│       ├── RF_Results.png
│       └── ROC_Curve_Rresult.png
├── logs
│   └── churn_library.log
├── models
│   ├── Logistic_Model.pkl
│   └── RFC_Model.pkl
├── LICENSE
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
└── requirements_py3.9.txt
```

There are four main files in this repository, including this one:

1. `churn_library.py` : Contains a model class with methods to perform the main steps of the ML workflow: import data, perform eda, feature engineering, training models, and the generation of some results plots.
2. `churn_notebook.ipynb`: A Jupyter notebook to run the ML workflow interactively.
3. `churn_script_logging_and_tests.py` : Used to test the functions of the ML workflow with logging written to the file `churn_library.py`.
4. `requirements.txt` : Specifies the libraries and dependencies to run this package.

There are also four folders in the repository:

1. `data`: location of data saved in `csv` format.
2. `images`: location where images are saved, specifically to subfolders named `eda` and `results`.
3. `logs` : location storing the log files of the testing results obtained from running `churn_script_logging_and_tests.py`.
4. `models`: location to store pickled models.


## Running Files

This ML workflow can be run interactively using the `churn_notebook.ipynb` or through the CLI with the following command:

```
ipython churn_library.py
```

### Dependencies

The list of dependencies required to run this package are as follows.

```
autopep8==1.6.0
joblib==1.1.0
matplotlib==3.5.2
numpy==1.21.5
pandas==1.4.4
pytest==7.1.2
pylint==2.14.5
scikit_learn==1.0.2
seaborn==0.11.2
shap==0.41.0
```

They can easily be installed using the following command:

```
pip install -r requirements.txt
```

### Testing and Logging

Testing any changes or customizations to the `churn_library.py` file can be done by running the `churn_script_logging_and_tests.py` through the command below.  This will generate a log of success, as well as errors to help with troubleshooting:

```
ipython churn_script_logging_and_tests.py
```

### Code Cleaning
This code complies with `PEP 8` rules. To check it automatically, run pylint on the terminal and see the score of each file:

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py


