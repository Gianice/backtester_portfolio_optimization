"""
datalayer.py
Loads, cleans, and validates historical market data.
No analysis, no strategy logic, no risk metrics — just trustworthy data out.
"""
import pandas as pd
import numpy as np


##-------1. load data --------##

def load_raw_data(filepath):
    raw_df = pd.read_excel(filepath, header = [0,1])
    return raw_df

##-------2. load clean_data_date --------##

def clean_data_date(raw_df):
    """
    Input : raw DataFrame
    Output: cleaned DataFrame —
            missing values handled, duplicates removed,
            sorted chronologically, consistent schema enforced
    """
    #-----------------Date-------------------#
    ##create new column to test check which date cannot be parsed successfully
    raw_df[('real_date','real_date')] = pd.to_datetime(raw_df[('Unnamed: 0_level_0','Dates')], errors = 'coerce')
    raw_df = raw_df.drop(('Unnamed: 0_level_0','Dates'), axis = 1)
    
    ##-------check any wrong date format--------##
    print(f'sum of NaN in dates = {raw_df[('real_date','real_date')].isna().sum()}')

    ##----------sort datetime----------#
    raw_df = raw_df.sort_values(('real_date', 'real_date'))
    
    ##------drop rows with duplicate date and ticker--------##
    raw_df = raw_df.drop_duplicates(subset = [('real_date', 'real_date')])
    
    ##-------set date as index---------#
    raw_df = raw_df.set_index(('real_date', 'real_date'))
    
    #-------Drop original date---------#
    raw_df = raw_df.drop(("Unnamed: 0_level_0","Dates"), axis = 1)
    
    return raw_df


##-------3. change_to_long data --------##
    
def change_to_long(raw_df):
    return raw_df.stack(level = 0)


##-------4. check type mismatch --------##

def check_type(long_raw_df):
    cols = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
    original_null = long_raw_df[cols].apply(lambda x: x.isnull().sum())
    
    long_raw_df[cols] = long_raw_df[cols].apply(pd.to_numeric, errors = 'coerce')
    
    after_null = long_raw_df[cols].apply(lambda x: x.isnull().sum())
    
    ##-----NaN difference between original and after--------#
    print(original_null - after_null)
    return long_raw_df
    


##-------5. drop non universal ticker --------##

def drop_non_universal(long_raw_df):
      #------------PRICE--------------#
    #1.check if it is under universal 100 in that year
    long_raw_df.index = long_raw_df.index.set_names(['real_date', 'ticker'])
    
    check_universal = long_raw_df.copy()

    check_universal = check_universal.reset_index()
    
    ##proportion of non-null/all < 0.5 -> drop
    check_universal = check_universal.drop('real_date', axis = 1).groupby('ticker').apply(lambda x: x.count()/(x.count()+x.isnull().sum())).map(lambda x: x<0.5)
    
    check_universal['to_drop'] = check_universal.any(axis = 1)
    
    to_drop_list = check_universal[check_universal['to_drop']].index.tolist()
    
    long_raw_df = long_raw_df[~long_raw_df.index.get_level_values('ticker').isin(to_drop_list)]
    
    return long_raw_df


##-------6. check misalign date --------##

def check_misalign_date(long_raw_data):
    long_raw_data = long_raw_data.reset_index()

    date_counts = long_raw_data.groupby('real_date')['ticker'].count()
    correct_date =  date_counts[date_counts == date_counts.max()]
    
    correct_date = correct_date.index.tolist()
    
    long_raw_data = long_raw_data[long_raw_data['real_date'].isin(correct_date)]
    
    long_raw_data = long_raw_data.set_index(['real_date', 'ticker'])
    
    return long_raw_data


##-------7. track unusual price and volume --------##

def check_price_and_volume(long_raw_df):
    # print(long_raw_df)
    long_raw_df.loc[lambda x: ~((x['PX_LOW'] < x['PX_LAST']) & (x['PX_LAST']  < x['PX_HIGH']) & (x['PX_LOW'] < x['PX_OPEN']) & (x['PX_OPEN']< x['PX_HIGH'])), ['PX_LOW', 'PX_HIGH', 'PX_OPEN', 'PX_LAST']] = None
    ##---------Track unusual volume with Z-score------------##
    
    mean = long_raw_df.groupby('ticker')['PX_VOLUME'].transform('mean')
    std = long_raw_df.groupby('ticker')['PX_VOLUME'].transform('std')
    long_raw_df['z_score'] = (long_raw_df['PX_VOLUME'] - mean) / std
    long_raw_df.loc[lambda x: x['z_score'].abs() > 3, 'PX_VOLUME'] = None
    long_raw_df = long_raw_df.drop('z_score', axis = 1)
    
    long_raw_df = long_raw_df.ffill(limit = 3)
    
    return long_raw_df

##-------8. add return columns --------##
def add_return_columns(aligned_df):
    """
    Input : aligned price DataFrame
    Output: same DataFrame + simple return and log return columns
            (basic derived series only — no risk/strategy metrics here)
    """
    aligned_df['simple_return'] = aligned_df.groupby('ticker')['PX_LAST'].transform('pct_change')
    
    aligned_df['log_return'] = aligned_df.groupby('ticker')['PX_LAST'].transform(lambda x : np.log(x/x.shift(1)))
    
    aligned_df[['simple_return', 'log_return']] = aligned_df[['simple_return', 'log_return']].fillna(0)
    
    return aligned_df



def validate_data(final_df):
    """
    Input : fully processed DataFrame
    Output: pass/fail or list of issues —
            checks for leftover NaNs, mismatched row counts,
            out-of-range values, unexpected date gaps
    """
    issues = []
    
    null_counts = final_df.isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"NaNs remaining (expected due to ffill limit=3):\n{null_counts[null_counts > 0]}")
    
    row_counts = final_df.groupby('ticker').size()
    if row_counts.nunique() > 1:
        issues.append(f"Mismatched row counts:\n{row_counts}")
    
    if (final_df[['PX_LAST','PX_OPEN','PX_HIGH','PX_LOW']] < 0).any().any():
        issues.append("Negative prices found")
    
    if issues:
        for i in issues:
            print(i)
    else:
        print("All checks passed")
        
        


clean_data(load_raw_data('dummy_data.xlsx'))