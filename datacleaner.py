from dataset import Dataset
import pandas as pd 
import numpy as np
class DataCleaner:
    def __init__(self, dataset, verbose:bool=True):
        self.dataset=dataset
        self.df = self.dataset.data.copy()
        self.verbose=verbose


    def log(self,msg):
        if self.verbose:
            print(f'INFO: {msg}')

    def column_names(self):
        self.df.columns=self.df.columns.str.strip().str.lower().str.replace(' ','_')
        self.log('Standardized column names')

    def drop_duplicates(self):
        dup_count=self.df.duplicated().sum()
        if dup_count>0:
            self.df.drop_duplicates(inplace=True)
            self.log(f'Removed {dup_count} duplicate rows')

    def string_data(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col]=self.df[col].astype('str').str.strip().str.lower()
        self.log('Standardized string columns')

    def null_values(self):
        placeholder_values = ['n/a', 'na', '--', '-', 'none', 'null', '', 'nan']
        self.df.replace(placeholder_values,np.nan, inplace=True)
        null_report=self.df.isnull().sum()
        null_report = null_report[null_report > 0]
        if not null_report.empty:
            self.log(f"Missing values found in columns:\n{null_report}")

    def constant_columns(self):
        constant_cols=[col for col in self.df.columns if self.df[col].nunique()==1]
        if constant_cols:
           self.log(f'Constant columns (consider removing): {constant_cols}')

    def encoding(self):
        high_card_cols=[col for col in self.df.select_dtypes(include='object') if self.df[col].nunique()>100]
        if high_card_cols:
            self.log(f"High-cardinality columns (consider encoding strategies): {high_card_cols}")

    def categorical_data(self):
        for col in self.df.select_dtypes(include='object'):
            n_unique= self.df[col].nunique()
            if n_unique< len(self.df)*0.05:
                self.df[col]=self.df[col].astype('category')
        self.log("Converted suitable object columns to category dtype.")

    def outliers(self):
        num_cols= self.df.select_dtypes(include=np.number).columns
        outlier_report={}
        for col in num_cols:
            q1, q3=self.df[col].quantile([0.25,0.75])
            iqr=q3-q1
            lower=q1-1.5*iqr
            upper=q3+1.5*iqr
            outliers=self.df[(self.df[col]<lower)|(self.df[col]>upper)][col].count()
            if outliers>0:
                outlier_report[col]=outliers
        if outlier_report:
            self.log(f"Potential numeric outliers detected:\n{outlier_report}")

    def clean(self):
        self.column_names()
        self.drop_duplicates()
        self.string_data()
        self.null_values()
        self.constant_columns()
        self.encoding()
        self.categorical_data()
        self.outliers()
        self.log("Data cleaning complete.")
        return self.df

dataset= Dataset.from_file('loan-recovery.csv')
clean_data= DataCleaner(dataset)
print(clean_data.df.head())
print(clean_data.clean())