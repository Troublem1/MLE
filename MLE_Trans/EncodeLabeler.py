import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

class MLETransformer(BaseEstimator, TransformerMixin):
        '''
        Allows the user enter a set of columns with categorical entries and preprocesses them into numeric values
        Args:
            cols: list of column names 
        Returns:
            categorical entries and preprocesses them into numeric values.
        '''

    def __init__ (self, cols):
        self.cols = cols
        self.codedict = {} # dict to take codes
        self.labeldict = {}   # dict to take coded representations
        self.my_labeler = LabelEncoder()  #initialize labelEncoder in class to save
    
    def fit(self, X , y=None):
        '''
        returns self of X
        Args:
            X: labels dataframe 
        Returns:
            self 
        '''
        self.cols = [eachcolumn for eachcolumn in self.cols if eachcolumn in X.columns]
        return self
    
    def transform(self, X):
        '''
        returns self of X
        Args:
            X: labels dataframe 
        Returns:
            Transformed Dataframe
        '''
        Xt = X.copy()
        for i in self.cols:
            self.labeldict[i] = list(Xt[i].unique())
            Xt[i] = self.my_labeler.fit_transform(Xt[i])
            self.codedict[i] = list(Xt[i].unique())
        return Xt
    
    def _display(self , key='key'):
        
        '''Display's a dict of codes and labels
        Args:
                key : column label
    
        returns:
                returns a dictionary of quantitative codes and their original labels
            '''
        print(dict(zip(self.codedict[key],self.labeldict.get(key))))