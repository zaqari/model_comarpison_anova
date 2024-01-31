from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Union, List

class ANOVA():

    def __init__(self, df: pd.DataFrame, group_column: str, y_column: str):
        super(ANOVA,self).__init__()
        self.group = group_column
        self.Y = y_column
        self.df = df
        self.df['e'] = None
        self.group_index_dic = {n: i for i, n in enumerate(self.df[self.group].unique())}

        # variables
        self.grand_mu = self.df[self.Y].astype(float).values.mean()
        self.alpha = np.array(
            [self.df[self.Y].loc[self.df[self.group].isin([group])].astype(float).mean() - self.grand_mu for group in
             self.df[self.group].unique()])

        self.__fit()

    def __fit(self):
        X = np.zeros(shape=(len(self.df), len(self.group_index_dic)))
        for i,v in enumerate(self.df[self.group].values):
            X[i,self.group_index_dic[v]] = 1.

        X = (X@self.alpha.reshape(-1,1)).reshape(-1)
        y_ = (X + self.grand_mu).reshape(-1)

        self.df['e'] = (self.df[self.Y].values - y_) * (X.reshape(-1) != 0).astype(float)
        self.resid = (self.df['e'].values**2).sum()
        self.msw = self.resid/(len(self.df) - len(self.alpha))

    def F(self, contrast: np.ndarray = np.array([])):
        c = contrast
        if len(contrast) < len(self.alpha):
            c = np.array([1] * len(self.alpha))
            c[-1] = -(len(c)-1)

        mu = self.grand_mu + self.alpha
        n = np.array([self.df[self.group].isin([group]).sum() for group in self.df[self.group].unique()])

        num = (mu @ c)**2
        denom = ((c**2)/n).sum()

        return ((num/denom) / self.msw)

