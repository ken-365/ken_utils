import numpy as np
import pandas as pd
import glob
import os, sys

class pd_read:
    def __init__(self):
        pass
    def read_bulkcsv(file_path, **kwargs):
        """
        Read all csv in file path to pandas
        Implementation: df = read_bulkcsv("data/to_path/train")
        """
        all_files = glob.glob(os.path.join(file_path, "*.csv"))
        df_from_each_file = (pd.read_csv(f) for f in all_files)
        return  pd.concat(df_from_each_file, ignore_index=True)

# dataframe edx
class eda_df:
    def __init__(self):
        pass
    def cshape(df):
        print(f'rows/columns: {df.shape[0]:,}/{df.shape[1]:,}')
        pass
    # check for imbalance 
    def find_propotion(feature,df):
        """
        Description: check propotion of categorical variable in a feature
        Input: a 'name' of the feature and name of df
        Output : dataframe
        Implementation : find_pp('feature',df)
        """
        return pd.DataFrame(df[feature].value_counts()).merge(pd.crosstab(df[feature],'Propotion',normalize='columns'),left_index=True,right_index=True)
    
    def propotion_on_binary(df,feature):
        """
        Description: check propotion of a categorical variable that is binary (buy or not (0,1))
        Input: df and a 'name' of the feature and name of df
        Output : dataframe
        Implementation : 
                test_ctrl_by_os = user.groupby('OS')['test'].apply(calc_ratios).unstack()
                test_ctrl_by_os
        """
        d = df.value_counts(normalize=True)
        d['total'] = df.shape[0]
        return d
    
    
    def df_unique_values(sub):
        """
        Description: return info about number of unique values in feature
        Input: dataframe
        Output : dataframe
        Implementation : df_unique_values(df)
        """
        indx = []
        colm1 = []
        colm2 = []
        for feature in sub.columns:
            indx.append(feature)
            colm1.append(sub[feature].nunique())
            colm2.append(list(sub[feature].unique()[:8]))
        pd.set_option('max_colwidth',200)
        pd.DataFrame({'no. unique value' :colm1,
                      'first 8 unique values': colm2},
                    index= indx)
        
    def df_unique_values2(sub):
        """
        Description: return info about number of unique values in feature
        Input: dataframe
        Output : text
        Implementation : df_unique_values2(df)
        """
        for feature in sub.columns:
            print (f'{feature:28s}: {sub[feature].nunique()} {list(sub[feature].unique()[:15])}')