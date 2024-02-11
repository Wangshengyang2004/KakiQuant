"""
update:2023/12/23
author:Nick_Ni
"""

import pandas as pd 
import numpy as np
from scipy import stats
import statsmodels.api as sm
from tqdm import *

from rqdatac import *
from rqfactor import *
from rqfactor.notebook import *
from rqfactor.extension import *
init()
import rqdatac

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 动态券池
def INDEX_FIX(start_date,end_date,index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str 
    :param index_item: 指数代码 -> str 
    :return index_fix: 动态因子值 -> unstack
    """
    
    index = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in index_components(index_item,start_date= start_date,end_date=end_date).items()])).T

    # 构建动态股票池 
    index_fix = index.unstack().reset_index().iloc[:,-2:]
    index_fix.columns = ['date','stock']
    index_fix.date = pd.to_datetime(index_fix.date)
    index_fix['level'] = True
    index_fix.dropna(inplace = True)
    index_fix = index_fix.set_index(['date','stock']).level.unstack()
    index_fix.fillna(False,inplace = True)

    return index_fix

# 再次定义函数：计算最大回撤
def get_Performance_analysis(T,year_day = 252):
    # 获取最终净值
    net_values = round(T[-1],4)
    
    # 计算几何年化收益率
    year_ret_sqrt = net_values**(year_day/len(T))-1
    year_ret_sqrt = round(year_ret_sqrt*100,2)
    
    # 计算年化波动率
    volitiy = T.pct_change().dropna().std()*np.sqrt(year_day)
    volitiy = round(volitiy*100,2)
    
    #计算夏普，无风险收益率记3%
    sharpe = (year_ret_sqrt - 3)/volitiy
    sharpe = round(sharpe,2)

    # 计算最大回撤
    # 最大回撤结束点
    i = np.argmax((np.maximum.accumulate(T) - T)/np.maximum.accumulate(T))
    # 开始点
    j = np.argmax(T[:i])

    downlow = round((1-T[i]/T[j])*100,2)

    # 输出
    return [net_values,year_ret_sqrt,sharpe,downlow,volitiy]
#------------------------------------------------------------------------

# 券池过滤
def get_new_stock_filter(stock_list,date_list, newly_listed_threshold = 252):

    listed_date_list = [rqdatac.instruments(stock).listed_date for stock in stock_list]        
    newly_listed_window = pd.Series(index=stock_list, data=[rqdatac.get_next_trading_date(listed_date, n=newly_listed_threshold) for listed_date in listed_date_list]) 
    # 
    newly_listed_window.index.names = ['order_book_id']
    newly_listed_window = newly_listed_window.to_frame('date')
    newly_listed_window['signal'] = True
    newly_listed_window = newly_listed_window.reset_index().set_index(['date','order_book_id']).signal.unstack('order_book_id').reindex(index=date_list)
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False)

    print('剔除新股已构建')

    return newly_listed_window

def get_st_filter(stock_list,date_list):
    st_filter = rqdatac.is_st_stock(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index = date_list)                                #剔除ST
    st_filter = st_filter.shift(-1).fillna(method = 'ffill')
    print('剔除ST已构建')

    return st_filter

def get_suspended_filter(stock_list,date_list):
    suspended_filter = rqdatac.is_suspended(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index=date_list)
    suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')
    print('剔除停牌已构建')

    return suspended_filter

def get_limit_up_down_filter(stock_list,date_list):

    # 涨停则赋值为1,反之为0    
    price = rqdatac.get_price(stock_list,date_list[0],date_list[-1],adjust_type='none',fields = ['open','limit_up'])
    df = (price['open'] == price['limit_up']).unstack('order_book_id').shift(-1).fillna(False)
    print('剔除开盘涨停已构建')

    return df

# 数据清洗函数 -----------------------------------------------------------
# MAD:中位数去极值
def mad(df):
    # MAD:中位数去极值
    def filter_extreme_MAD(series,n): 
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n*new_median,median + n*new_median)
    # 离群值处理
    df = df.apply(lambda x :filter_extreme_MAD(x,3), axis=1)

    return df

def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

import os

def create_dir_not_exist(path):
    # 若不存在该路径则自动生成
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def neutralization(df,order_book_ids,index_item = ''):

    """
    :param df: 因子值 -> unstack
    :param df_result: 中性化后的因子值 -> unstack
    """

    # order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime("%F")
    end = datetime_period[-1].strftime("%F")
    #获取行业/市值暴露度
    try:
        df_industy_market = pd.read_pickle(f'tmp/df_industy_market_{index_item}_{start}_{end}.pkl')
    except:
        market_cap = execute_factor(LOG(Factor('market_cap_3')),order_book_ids,start,end).stack().to_frame('market_cap')
        industry_df = get_industry_exposure(order_book_ids,datetime_period)
        #合并因子
        industry_df['market_cap'] = market_cap
        df_industy_market = industry_df
        df_industy_market.index.names = ['datetime','order_book_id']
        df_industy_market.dropna(inplace = True)
        create_dir_not_exist('tmp')
        df_industy_market.to_pickle(f'tmp/df_industy_market_{index_item}_{start}_{end}.pkl')

    df_industy_market['factor'] = df.stack()
    df_industy_market.dropna(subset = 'factor',inplace = True)
    
    #OLS回归
    df_result = pd.DataFrame(columns = order_book_ids,index = datetime_period)
    for i in tqdm(datetime_period):
        try:
            df_day = df_industy_market.loc[i]
            x = df_day.iloc[:,:-1]   #市值/行业
            y = df_day.iloc[:,-1]    #因子值
            df_result.loc[i] = sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid
        except:
            pass
    df_result.index.names = ['datetime']

    return df_result


def get_industry_exposure(order_book_ids,datetime_period):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 虚拟变量 -> dataframe
    """
    print('gen industry martix... ')
    zx2019_industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
    df = pd.DataFrame(zx2019_industry)
    df.set_index(['order_book_id', 'start_date'], inplace=True)
    df = df['first_industry_name'].sort_index()
    
    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'datetime'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()
    
    #生成行业虚拟变量
    return pd.get_dummies(result)



def data_clean(df,index_fix,index_item = ''):
    stock_list = index_fix.columns.tolist()
    start_date = index_fix.index[0].strftime('%F')
    end_date = index_fix.index[-1].strftime('%F')
    date_list = index_fix.index.tolist()
    try:
        combo_mask = pd.read_pickle(f'tmp/combo_mask_{index_item}_{start_date}_{end_date}.pkl')
    except:
        new_stock_filter = get_new_stock_filter(stock_list,date_list)
        st_filter = get_st_filter(stock_list,date_list)
        suspended_filter = get_suspended_filter(stock_list,date_list)
        limit_up_down_filter = get_limit_up_down_filter(stock_list,date_list)

        combo_mask = (new_stock_filter.astype(int) 
                    + st_filter.astype(int)
                    + suspended_filter.astype(int)
                    + limit_up_down_filter.astype(int)
                    + (~index_fix).astype(int)) == 0

        create_dir_not_exist('tmp')
        combo_mask.to_pickle(f'tmp/combo_mask_{index_item}_{start_date}_{end_date}.pkl')
    
    df = df.mask(~combo_mask).dropna(how = 'all',axis = 1)
    df = standardize(neutralization(standardize(mad(df)),stock_list,index_item))
    df = df.apply(lambda x: x.astype(float))
    
    return df

# 单因子检测函数 -----------------------------------------------------------

# IC计算  
def Quick_Factor_Return_N_IC(df,n,index_item,name = '',Rank_IC = True):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param index_item: 券池 -> str
    :param name: 因子名称 -> str
    :param True/False: Rank_ic/Normal_ic -> bool
    :return result: ic序列 -> series
    :return report: ic报告 -> dataframe
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%F')
    end = datetime_period[-1].strftime('%F')
    try:
        close = pd.read_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')
    except:
        index_fix = INDEX_FIX(start,end,index_item)
        order_book_ids = index_fix.columns.tolist()
        close = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id')
        create_dir_not_exist('tmp')
        close.to_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')
    
    return_n = close.pct_change(n).shift(-n)

    if Rank_IC == True:
        result = df.corrwith(return_n,axis = 1,method='spearman').dropna(how = 'all')
    else:
        result = df.corrwith(return_n,axis = 1,method='pearson').dropna(how = 'all')
    
    t_stat,_ = stats.ttest_1samp(result, 0)
    
    report = {'name': name,
    'IC mean':round(result.mean(),4),
    'IC std':round(result.std(),4),
    'IR':round(result.mean()/result.std(),4),
    'IR_ly':round(result[-252:].mean()/result[-252:].std(),4),
    'IC>0':round(len(result[result>0].dropna())/len(result),4),
    'ABS_IC>2%':round(len(result[abs(result) > 0.02].dropna())/len(result),4),
    't_stat':round(t_stat,4),
    }
    
    print(report)
    report = pd.DataFrame([report])
    
    return result,report


def factor_ret_tvalue(df,n,index_item,name = ''):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param index_item: 券池 -> str
    :param name: 因子名称 -> str
    :return result: tvalue值 -> dataframe
    :return report: tvalue报告 -> dataframe
    """
    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%F')
    end = datetime_period[-1].strftime('%F')
    try:
        close = pd.read_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')
    except:
        index_fix = INDEX_FIX(start,end,index_item)
        order_book_ids = index_fix.columns.tolist()
        close = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id')
        create_dir_not_exist('tmp')
        close.to_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')
    
    return_n = close.pct_change(n).shift(-n).dropna(how = 'all')
    inter_index = sorted(set(return_n.index) & set(df.index))
    result = {}
    for t in inter_index:
        y = return_n.loc[t].dropna()
        x = df.loc[t].dropna()
        inter_stock = sorted(set(x.index) & set(y.index))
        y = y.loc[inter_stock]
        x = x.loc[inter_stock]
        regression = sm.OLS(y,sm.add_constant(x)).fit()
        result[t] = [regression.tvalues[1],regression.pvalues[1]]

    result = pd.DataFrame(result,index = ['t_values','p_values']).T

    report = {'name': name,
    'Tvalue mean':round(result['t_values'].abs().mean(),4),
    'Pvalue mean':round(result['p_values'].mean(),4),
    'pvalue_abs < 1%':round(len(result['p_values'][result['p_values'] < 0.01].dropna())/len(result),4)
    }

    print(report)
    report = pd.DataFrame([report])

    return result,report

# 分层效应

def group_g(df,n,g,index_item):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param g: 分组数量 -> int 
    :return group_return: 各分组日收益率 -> dataframe
    :return turnover_ratio: 各分组日调仓日换手率 -> dataframe
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%F')
    end = datetime_period[-1].strftime('%F')

    try:
        return_1d = pd.read_pickle(f'tmp/return_1d_{index_item}_{start}_{end}.pkl')
    except:
        index_fix = INDEX_FIX(start,end,index_item)
        order_book_ids = index_fix.columns.tolist()
        return_1d = get_price(order_book_ids,get_previous_trading_date(start,1,market='cn'),end,
                                '1d','close','pre',False,True).close.unstack('order_book_id').pct_change().shift(-1).dropna(axis = 0,how = 'all').stack()
        create_dir_not_exist('tmp')
        return_1d.to_pickle(f'tmp/return_1d_{index_item}_{start}_{end}.pkl')

    group = df.stack().to_frame('factor')
    group['current_renturn'] = return_1d
    group = group.dropna()
    group.reset_index(inplace = True)
    group.columns = ['date','stock','factor','current_renturn']

    turnover_ratio = pd.DataFrame()
    group_return = pd.DataFrame()

    for i in range(0,len(datetime_period)-1,n):  # -1 防止刚好切到最后一天没法计算
        # 截面分组
        single = group[group.date == datetime_period[i]].sort_values(by = 'factor')
        single.loc[:,'group'] = pd.qcut(single.factor,g, list(range(1,g+1))).to_list()
        group_dict = {}
        for j in range(1,g+1):
            group_dict[j] = single[single.group == j].stock.tolist()
        
        # 计算换手率
        turnover_ratio_temp = []
        if i == 0:
            temp_group_dict = group_dict
        else:
            for j in range(1,g+1):
                turnover_ratio_temp.append(len(list(set(temp_group_dict[j]).difference(set(group_dict[j]))))/len(set(temp_group_dict[j])))
            turnover_ratio = pd.concat([turnover_ratio,
                                        pd.DataFrame(turnover_ratio_temp,
                                                    index = ['G{}'.format(j) for j in list(range(1,g+1))],
                                                    columns = [datetime_period[i]]).T],
                                                    axis = 0)
            temp_group_dict = group_dict
        
        # 获取周期
        if i < len(datetime_period)-n:
            period = group[group.date.isin(datetime_period[i:i+n])]
        else:
            period = group[group.date.isin(datetime_period[i:])]

        # 计算各分组收益率
        group_return_temp = []
        for j in range(1,g+1):
            group_return_temp.append(period[period.stock.isin(group_dict[j])].set_index(['date','stock']).current_renturn.unstack('stock').mean(axis = 1))
        group_return = pd.concat([group_return,pd.DataFrame(group_return_temp,index = ['G{}'.format(j) for j in list(range(1,g+1))]).T],axis = 0)
        print('\r 当前：{} / 总量：{}'.format(i,len(datetime_period)),end='')
    
    return group_return,turnover_ratio