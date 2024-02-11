import pandas as pd 
import numpy as np
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm



def neutralization(factor,market_cap = market_cap,industry_exposure = industry_exposure):
    factor_resid = pd.DataFrame()
    factor_ols = pd.concat([factor.stack(),market_cap,industry_exposure],axis = 1).dropna()
    for i in datetime_list:
        factor_ols_temp = factor_ols.loc[i]    #截面数据做回归
        x = factor_ols_temp.iloc[:,1:]   #市值/行业
        y = factor_ols_temp.iloc[:,0]    #因子值
        factor_ols_resid_temp = pd.DataFrame(sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid,columns = ['{}'.format(i)])
        factor_resid = pd.concat([factor_resid,factor_ols_resid_temp],axis = 1)
    factor_resid = factor_resid.T
    factor_resid.index = pd.to_datetime(factor_resid.index)
    return factor_resid

def Factor_Return_N_IC(factor,n,Rank_IC = True,close = close):

    date_list_whole = sorted(list(set(factor.index.get_level_values(0))))
    start_date = date_list_whole[0]
    end_date = date_list_whole[-1]
    stock_list = sorted(list(set(factor.index.get_level_values(1))))
    close = close.pct_change(n).shift(-n).stack()
    close = pd.concat([close,factor],axis =1).dropna().reset_index()
    close.columns = ['date','stock','change_days','factor']
    if Rank_IC == True:
        rank_ic = close.groupby('date')['change_days','factor'].corr(method = 'spearman').reset_index().set_index(['date'])
    
    return rank_ic[rank_ic.level_1 == 'factor'][['change_days']]


def ic_ir(x,factor_name):
    t_stat, p_value = stats.ttest_1samp(x, 0)
    IC = {'name': factor_name,
        'IC mean':round(x.mean()[0],4),
        'IC std':round(x.std()[0],4),
        'IR':round(x.mean()[0]/x.std()[0],4),
        't_stat':round(t_stat[0],4),
        'p_value':round(p_value[0],4),
        'IC>0':round(len(x[x>0].dropna())/len(x),4),
        'ABS_IC>2%':round((len(x[abs(x) > 0.02].dropna())/len(x)),4)}
    return pd.DataFrame([IC])