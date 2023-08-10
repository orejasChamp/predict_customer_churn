"""
Functions for predicting customer churn.

Author: Rachel Guha
Date: 08 August 2023
"""

# import libraries
import random
import warnings
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
sb.set()


# Set up OS.
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Suppress warnings.
warnings.filterwarnings('ignore')

# Set a random seed.
random.seed(42)

# Setting display to show all rows.
pd.options.display.max_rows = 260

# Set-up colours to use in visualizations.
plt.style.use("seaborn-white")
palette = sb.color_palette("viridis")

color1 = palette[0]
color2 = palette[1]
color3 = palette[2]
color4 = palette[3]
color5 = palette[4]
color6 = palette[5]

# Setup parameters to standardize the visualizations.

# Axes.
rcParams['axes.spines.bottom'] = True
rcParams['axes.spines.left'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.grid'] = True
rcParams['axes.grid.axis'] = 'y'
rcParams['grid.color'] = 'lightgrey'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.axisbelow'] = True
rcParams['axes.linewidth'] = 2
rcParams['axes.ymargin'] = 0
rcParams['axes.labelsize'] = 16

# Legend.
rcParams['legend.fontsize'] = 8

# Ticks and tick labels.
rcParams['axes.edgecolor'] = 'grey'
rcParams['xtick.color'] = 'grey'
rcParams['ytick.color'] = 'grey'
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 0
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 0
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

# Fonts.
rcParams['font.size'] = 14
rcParams['font.family'] = 'serif'
rcParams['text.color'] = 'grey'
rcParams['axes.labelcolor'] = 'grey'


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Plot RF classification report.
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/RF_Results.png',
                bbox_inches='tight')
    plt.close()

    # Plot LR classification report.
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/Logistic_Results.png',
                bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_train):
    '''
    creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values

    output:
             None
    '''

    # Calculate feature importances.
    importances = model.feature_importances_

    # Sort feature importances in descending order.
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances.
    names = [X_train.columns[i] for i in indices]

    # Create plot.
    plt.figure(figsize=(15, 5))
    plt.title("Feature Importance")
    plt.ylabel('importance')
    plt.bar(range(X_train.shape[1]), importances[indices], color=color4)
    #plt.grid(b = False, axis = 'y')
    plt.xticks(range(X_train.shape[1]), names, rotation=90)
    plt.savefig('./images/results/Feature_Importances.png',
                bbox_inches='tight')
    plt.close()


class Model:
    """
    A class to represent a model for supervised machine learning.

    ...

    Attributes
    ----------
    df : data type,
        description
    X_train : data type,
        description
    y_train : data type,
        description
    X_test : data type,
        description
    y_test : data type,
        description

    Methods
    -------
    import_data:
        returns dataframe for the csv found at pth
    perform_eda:
        perform eda on df and save figures to images folder
    encoder_helper:
        helper function to turn each categorical column into a
        new column with propotion of churn for each category
    perform_feature_engineering:
        split data into train and test sets
    train_models:
         train, store model results: images + scores, and store models
    classification_report_image:
        produces classification report for training and testing results
        and stores report as image in images folder

    """

    def __init__(self):
        """
        Initializes all the necessary attributes for the model object.

        """

        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
            pth: a path to the csv

        output:
            df: pandas dataframe
        '''
        df = pd.read_csv(pth)

        # Calculate churn label.
        df['Churn'] = df['Attrition_Flag']\
            .apply(lambda val: 0 if val == "Existing Customer" else 1)

        # Drop unneeded columns.
        self.df = df.drop(
            ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)

        return self.df

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder

        input:
            df: dataframe

        output:
            None
        '''

        # Group columns by data type, categorical and quantitative.
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        quant_cols = self.df.select_dtypes(include='number').columns.tolist()

        # Create and save histograms of quantitative variables.
        for col in quant_cols:
            fig, ax = plt.subplots(figsize=(12, 8))
            self.df[col].hist(color=color4).grid(axis='x')
            plt.title(f'Distribution of {col}', fontsize=20)
            plt.xlabel("column bins", fontsize=14)
            plt.ylabel("count", fontsize=14)
            ax.set_axisbelow(True)
            plt.savefig(f'./images/eda/{col}.png')
            # plt.show()
            plt.close()

        # Create and save histograms of categorical variables.
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(12, 8))
            self.df[col].value_counts('normalize').plot(
                kind='bar', color=palette, rot=45)
            plt.title(f'Proportion of {col}', fontsize=20)
            plt.ylabel("proportion", fontsize=14)
            ax.set_axisbelow(True)
            plt.savefig(f'./images/eda/{col}.png')
            # plt.show()
            plt.close()

        # Create and save a heatmap.
        fig, ax = plt.subplots(figsize=(12, 8))
        sb.heatmap(self.df.corr(), annot=False, cmap='viridis_r', linewidths=2)
        plt.savefig('./images/eda/Heatmap.png')
        # plt.show()
        plt.close()

        print('Plots generated and saved to the images/eda folder.')

    def encoder_helper(self, response):
        '''
        helper function to turn each categorical column into a
        new column with propotion of churn for each category

        input:
            df: dataframe
            category_lst: list of columns that contain categorical features

        output:
            df: pandas dataframe with new encoded columns
        '''

        cat_cols = self.df.select_dtypes(include='object').columns.tolist()

        for col in cat_cols:
            new_col = col + '_' + response
            cat_grp = self.df.groupby(col).mean()[response]
            self.df[new_col] = self.df[col].apply(lambda x: cat_grp.loc[x])
            self.df.drop([col], axis=1, inplace=True)

        print('Encoding of categorical variables completed.')

        return self.df

    def feature_engineering(self, response, test_size=0.3):
        '''
        creates and stores the feature importances in pth

        input:
              df: pandas dataframe

        output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
        '''

        X = self.df.drop([response], axis=1)
        X_col_names = X.columns
        y = self.df[response]

        # Apply Standard Scaler to features.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(scaled, columns=X_col_names)

        # Split data.
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        print('Data scaled and split into training and test sets.')
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        '''
        train, store model results: images + scores, and store models

        input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
        output:
              None
        '''

        # Initialize ML models.
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        # Set-up grid search for RFC training optimization.
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        # Train RF model with grid search CV.
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)

        # Train LR model.
        lrc.fit(self.X_train, self.y_train)

        # Get RF predictions using best estimator.
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

        # Get LR predictions using best estimator.
        y_train_preds_lr = lrc.predict(self.X_train)
        y_test_preds_lr = lrc.predict(self.X_test)

        # Save best models.
        joblib.dump(cv_rfc.best_estimator_, './models/RFC_Model.pkl')
        joblib.dump(lrc, './models/Logistic_Model.pkl')
        print('Model training complete.  Saving best models to the models folder.')

        # Save classification reports.
        classification_report_image(self.y_train,
                                    self.y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)
        print('Classification reports generated and saved to the images/results folder.')

        # Save ROC curve.
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(lrc,
                       self.X_test,
                       self.y_test,
                       ax=ax,
                       color=color2)
        plot_roc_curve(cv_rfc.best_estimator_,
                       self.X_test,
                       self.y_test,
                       ax=ax,
                       alpha=0.8,
                       color=color4)
        plt.savefig('./images/results/ROC_Curve_Result.png')
        plt.close()
        print('ROC curves generated and saved to the images/results folder.')

        # Display feature importance.
        feature_importance_plot(cv_rfc.best_estimator_, self.X_train)
        print('Feature importance plot generated and saved to the images/results folder.')


if __name__ == "__main__":

    PATH = './data/bank_data.csv'
    RESPONSE = 'Churn'

    # Initialize model.
    MODEL = Model()

    # Import data.
    churn_model = MODEL.import_data(PATH)

    # Print info about dataset.
    MODEL.print_info()

    # Perform EDA.
    MODEL.perform_eda()

    # Encode categorical variables.
    churn_model = MODEL.encoder_helper(RESPONSE)

    # Perform feature engineering.
    churn_model = MODEL.feature_engineering(RESPONSE)

    # Train models and output results.
    MODEL.train_models()
