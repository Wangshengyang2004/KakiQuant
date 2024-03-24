
from hmmlearn.hmm import GMMHMM, GaussianHMM
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import *
import pymongo
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)




import csv
client = pymongo.MongoClient('mongodb://10.201.8.134/')
df = pd.DataFrame(list(client['crypto_db']['BTC-USDT-SWAP-1D'].find()))
df.to_csv('btc_data.csv')
import pandas as pd

def reverse_csv(input_file, output_file):
    pd.read_csv(input_file)[::-1].to_csv(output_file, index=False)

# Example usage
reverse_csv("btc_data.csv","rev.csv")



# ### 1 基础数据获取


csv_file = "rev.csv"
def run(csv_file = "rev.csv",date='2023-01-01',r0 = 0.1,hist_len0 = 200,retrain_gap0 = 60):
    csv_data = pd.read_csv(csv_file) 
    df = pd.DataFrame(csv_data)
    index_ret = df['close'].pct_change()
    df.head()





    # ### 2 特征构建


    train = df.loc[date:]
    test = df.loc[:date]
    train['mtm1'] = train['close'].pct_change()
    train['mtm5'] = train['close'].pct_change(5)
    train['diffreturn'] = (train['high'] - train['low'])/train['low']

    train = train.dropna()
    closeidx = train['close']
    datelist = pd.to_datetime(train.index)
    train_X = np.array(train[['mtm1','mtm5','diffreturn']])


    # ### 3 隐马尔科夫模型
    # #### 3.1 模型训练


    hmm = GaussianHMM(n_components=4, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
    latent_states_sequence = hmm.predict(train_X)


    # #### 3.2 训练分类结果


    # sns.set_style('white')
    plt.figure(figsize=(10, 5))

    for i in range(hmm.n_components):
        state = (latent_states_sequence == i)
        print(f'类型{i}数量：{len(datelist[state])}')
        # 作图
        plt.plot(datelist[state],                      # x坐标
                closeidx[state],                      # y坐标
                '.',                                  # 画图样式
                label=f'latent state {i}',          # 图示
                )
        plt.legend()
        plt.grid(1)

    data = pd.DataFrame({
        'datelist': datelist,
        'mtm1': train['mtm1'],
        'state': latent_states_sequence
    }).set_index('datelist')
    plt.savefig("../web/web/assets/training_result.jpg") ## 保存图片
    plt.show()




    # #### 3.3 收益分组


    plt.figure(figsize=(10, 5))
    for i in range(hmm.n_components):
        name = f'state {i}_return'
        state = (latent_states_sequence == i)
        idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
        data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
        plt.plot((data[name]+1).cumprod(),                     # 计算累计收益
                label=name)
        plt.legend()
        plt.grid(1)
    plt.savefig("../web/assets/profit_classification_result.jpg") ## 保存图片
    plt.show()
    data


    # #### 3.4 参数遍历


    for n_components in range(3,8):
        hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)
        plt.figure(figsize=(10, 5))
        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
            plt.plot((data[name]+1).cumprod(),                     # 计算累计收益
                    label=name)
            plt.legend()
            plt.grid(1)
            add = "../web/assets/parameters"+str(i-1)+".jpg"
        plt.savefig(add) ## 保存图片
        plt.show()


    # #### 3.5 目标函数设定：如何定义什么叫做好的分类


    cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
    cumnet


    # ##### 3.5.1 夏普加权计算


    # 夏普计算
    def get_sharpe(x):
        # 输入累计收益序列
        final_ret = x.iloc[-1]
        ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
        annual_ret = final_ret ** (252/len(ret)) - 1
        std_ret = ret.std()            
        sharpe = abs(annual_ret/std_ret)                    # 绝对值                

        return sharpe

    cumnet.apply(lambda x: get_sharpe(x))


    fitness_result = {}
    for n_components in range(3,8):
        print(n_components)
        hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)
        data = pd.DataFrame({
                                'datelist': datelist,
                                'mtm1': train['mtm1'],
                                'state': latent_states_sequence
                                }).set_index('datelist')
        
        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        fitness = cumnet.apply(lambda x: get_sharpe(x))
        print(fitness)
        fitness_result[n_components] = round(fitness.mean(),4)


    ax = pd.Series(fitness_result).plot(kind = 'bar')
    fig=ax.get_figure()
    fig.savefig('../web/assets/sharpratio_result.jpg')
    # plt.savefig("./front/web/assets/training_result.jpg") ## 保存图片
    # plt.show()


    # ##### 3.5.2 夏普样本加权计算


    # 夏普样本加权计算
    def get_sharpe_weight(x):
        # 输入累计收益序列
        final_ret = x.iloc[-1]
        ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
        annual_ret = final_ret ** (252/len(ret)) - 1
        std_ret = ret.std()            
        sharpe = abs(annual_ret/std_ret)                    # 绝对值  
        sharpe = sharpe * len(ret) / len(x)                 # 数据量修正

        return sharpe

    cumnet.apply(lambda x: get_sharpe_weight(x))


    fitness_result = {}
    for n_components in range(3,8):
        print(n_components)
        hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)
        data = pd.DataFrame({
                                'datelist': datelist,
                                'mtm1': train['mtm1'],
                                'state': latent_states_sequence
                                }).set_index('datelist')
        
        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        fitness = cumnet.apply(lambda x: get_sharpe_weight(x))
        print(fitness)
        fitness_result[n_components] = round(fitness.mean(),4)


    ax=pd.Series(fitness_result).plot(kind = 'bar')
    fig=ax.get_figure()
    fig.savefig('../web/assets/Sharpe_Sample_Weighted_Calculation_result.jpg')


    pd.Series(fitness_result).sort_values().index[-1]


    # #### 3.6 选入最优的参数


    train = df.loc[:date]

    train['mtm1'] = train['close'].pct_change()
    train['mtm5'] = train['close'].pct_change(5)
    train['diffreturn'] = (train['high'] - train['low'])/train['low']

    train = train.dropna()
    closeidx = train['close']
    datelist = pd.to_datetime(train.index)
    train_X = np.array(train[['mtm1','mtm5','diffreturn']])

    hmm = GaussianHMM(n_components=5, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
    latent_states_sequence = hmm.predict(train_X)

    data = pd.DataFrame({
        'datelist': datelist,
        'mtm1': train['mtm1'],
        'state': latent_states_sequence
    }).set_index('datelist')

    plt.figure(figsize=(10, 5))
    for i in range(hmm.n_components):
        name = f'state {i}_return'
        state = (latent_states_sequence == i)
        idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
        data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
        plt.plot((data[name]+1).cumprod(),                     # 计算累计收益
                label=name)
        plt.legend()
        plt.grid(1)
    plt.savefig("../web/assets/best_param.jpg") ## 保存图片
    plt.show()

    data


    # 策略: 清仓1、4信号，保留2，3信号 ，半仓0信号


    # ### 4 样本外训练


    test = df.loc[date:]

    test['mtm1'] = test['close'].pct_change()
    test['mtm5'] = test['close'].pct_change(5)
    test['diffreturn'] = (test['high'] - test['low'])/test['low']

    test = test.dropna()
    closeidx = test['close']
    datelist = pd.to_datetime(test.index)
    test_X = np.array(test[['mtm1','mtm5','diffreturn']])


    latent_states_sequence = hmm.predict(test_X)
    latent_states_sequence


    data = pd.DataFrame({
        'datelist': datelist,
        'mtm1': test['mtm1'],
        'state': latent_states_sequence
    }).set_index('datelist')

    plt.figure(figsize=(10, 5))
    for i in range(hmm.n_components):
        name = f'state {i}_return'
        state = (latent_states_sequence == i)
        idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
        data[name] = data.mtm1.multiply(idx, axis=0)      # 收益计入对应信号中去
        plt.plot((data[name]+1).cumprod(),                     # 计算累计收益
                label=name)
        plt.legend()
        plt.grid(1)
    plt.savefig("../web/assets/training_outside.jpg") ## 保存图片
    plt.show()
    data


    strategy = (
                data['state 0_return'] * 0.5
                + data['state 1_return'] * 0
                + data['state 2_return'] * 1
                + data['state 3_return'] * 1
                + data['state 4_return'] * 0
                )


    net = pd.concat([strategy.to_frame('strategy'),test['mtm1'].to_frame('benchmark')],axis = 1)
    print(strategy.to_frame('strategy'))
    print(test['mtm1'].to_frame('benchmark'))
    cumnet = (1 + net).cumprod()
    ax = cumnet.plot()
    fig=ax.get_figure()
    fig.savefig('../web/assets/predict.jpg')
    # fig.savefig('../front/pages/1.png')



    # ### 5 滚动训练


    csv_file = "rev.csv"
    csv_data = pd.read_csv(csv_file)  # 防止弹出警告
    df = pd.DataFrame(csv_data)
    index_ret = df['close'].pct_change()

    df.head()


    r= r0
    hist_len = hist_len0
    retrain_gap = retrain_gap0
    data_trade = []
    length = df.shape[0]

    # 夏普样本加权计算
    def get_sharpe_weight(x):
        # 输入累计收益序列
        try:                                                                      # 防止进来的就没有数据
            final_ret = x.iloc[-1]
            ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
            annual_ret = final_ret ** (252/len(ret)) - 1
            std_ret = ret.std()            
            sharpe = abs(annual_ret/std_ret)                                      # 绝对值  
            sharpe = sharpe * len(ret) / len(x)                                   # 数据量修正
        except:
            sharpe = 0

        return sharpe

    def get_param(train,train_X,datelist):
        fitness_result = {}
        for n_components in range(3,8):
            try:
                hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
                latent_states_sequence = hmm.predict(train_X)
                data = pd.DataFrame({
                                        'datelist': datelist,
                                        'mtm1': train['mtm1'],
                                        'state': latent_states_sequence
                                        }).set_index('datelist')
                
                for i in range(hmm.n_components):
                    name = f'state {i}_return'
                    state = (latent_states_sequence == i)
                    idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
                    data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
                cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
                fitness = cumnet.apply(lambda x: get_sharpe_weight(x))
                fitness_result[n_components] = round(fitness.mean(),4)
            except:
                continue
        best_n_components = pd.Series(fitness_result).sort_values().index[-1]

        return best_n_components


    def get_signal_return(latent_states_sequence,train,datelist):
        data = pd.DataFrame({
            'datelist': datelist,
            'mtm1': train['mtm1'],
            'state': latent_states_sequence
        }).set_index('datelist')

        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
        
        return data

    for i in tqdm(range(hist_len,length,retrain_gap)):

        # 1. 训练集特征构建
        train = df.iloc[i-hist_len:i]
        train['mtm1'] = train['close'].pct_change()
        train['mtm5'] = train['close'].pct_change(5)
        train['diffreturn'] = (train['high'] - train['low'])/train['low']
        train['volume_z'] = (train['volume'] - train['volume'].rolling(20).mean())/train['volume'].rolling(20).std()
        train = train.dropna()
        closeidx = train['mtm1']
        train_X = np.array(train[['mtm5','diffreturn','volume_z']])
        datelist = pd.to_datetime(train.index)

        # 2. 训练集参数遍历
        best_n_components = get_param(train,train_X,datelist)
        
        # 3. 训练集内放入最优参数
        
        hmm = GaussianHMM(n_components = best_n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)

        data = get_signal_return(latent_states_sequence,train,datelist)

        # 4. 选择清仓的信号
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        get_signal = (cumnet.iloc[-1] < r).replace(False,np.nan).dropna().index.tolist()
        get_signal = [int(i[6:7]) for i in get_signal]                     # 转成数字形式

        # 5. 样本外计算
        test = df.iloc[i - hist_len: i + retrain_gap - 1]       # 因为有预计算所以先长的算完，再切分 （往前20天也够了）
        test['mtm1'] = test['close'].pct_change()
        test['mtm5'] = test['close'].pct_change(5)
        test['diffreturn'] = (test['high'] - test['low'])/test['low']
        test['volume_z'] = (test['volume'] - test['volume'].rolling(20).mean())/test['volume'].rolling(20).std()
        test = test.dropna()
        test = test.iloc[-(retrain_gap - 1):]
        closeidx = test['mtm1']
        test_X = np.array(test[['mtm5','diffreturn','volume_z']])
        
        datelist = test.index
        print(datelist)
        
        latent_states_sequence = hmm.predict(test_X)

        data = pd.DataFrame({
                'datelist': datelist,
                'mtm1': test['mtm1'],
                'state': latent_states_sequence
            }).set_index('datelist')
        
        # 6. 清仓信号叠加
        data_trade += data[data.state.isin(get_signal)].index.tolist()


    # #### 5.1 收益计算


    net = index_ret.to_frame('ret').iloc[hist_len:]

    net.loc[data_trade,'clear_day'] = 0           # 加上清仓信号
    net = net.fillna(1)

    for i in [0,1]:
        name = i
        state = net.clear_day == i
        idx = np.append(0, state[:-1])
        net[name] = net.ret.multiply(idx, axis=0)

    cumnet = (net.drop(['ret','clear_day'],axis = 1) + 1).cumprod()
    ax = cumnet.plot()
    cumnet
    fig=ax.get_figure()
    fig.savefig('../web/assets/profit_calculation.jpg')


    # #### 5.2 阈值调整


    r= r0+0.1
    hist_len = hist_len0
    retrain_gap = retrain_gap0
    data_trade = []
    length = df.shape[0]

    # 夏普样本加权计算
    def get_sharpe_weight(x):
        # 输入累计收益序列
        try:                                                                      # 防止进来的就没有数据
            final_ret = x.iloc[-1]
            ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
            annual_ret = final_ret ** (252/len(ret)) - 1
            std_ret = ret.std()            
            sharpe = abs(annual_ret/std_ret)                                      # 绝对值  
            sharpe = sharpe * len(ret) / len(x)                                   # 数据量修正
        except:
            sharpe = 0

        return sharpe

    def get_param(train,train_X,datelist):
        fitness_result = {}
        for n_components in range(3,8):
            try:
                hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
                latent_states_sequence = hmm.predict(train_X)
                data = pd.DataFrame({
                                        'datelist': datelist,
                                        'mtm1': train['mtm1'],
                                        'state': latent_states_sequence
                                        }).set_index('datelist')
                
                for i in range(hmm.n_components):
                    name = f'state {i}_return'
                    state = (latent_states_sequence == i)
                    idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
                    data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
                cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
                fitness = cumnet.apply(lambda x: get_sharpe_weight(x))
                fitness_result[n_components] = round(fitness.mean(),4)
            except:
                continue
        best_n_components = pd.Series(fitness_result).sort_values().index[-1]

        return best_n_components


    def get_signal_return(latent_states_sequence,train,datelist):
        data = pd.DataFrame({
            'datelist': datelist,
            'mtm1': train['mtm1'],
            'state': latent_states_sequence
        }).set_index('datelist')

        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
        
        return data

    for i in tqdm(range(hist_len,length,retrain_gap)):

        # 1. 训练集特征构建
        train = df.iloc[i-hist_len:i]
        train['mtm1'] = train['close'].pct_change()
        train['mtm5'] = train['close'].pct_change(5)
        train['diffreturn'] = (train['high'] - train['low'])/train['low']
        train['volume_z'] = (train['volume'] - train['volume'].rolling(20).mean())/train['volume'].rolling(20).std()
        train = train.dropna()
        closeidx = train['mtm1']
        train_X = np.array(train[['mtm5','diffreturn','volume_z']])
        datelist = pd.to_datetime(train.index)

        # 2. 训练集参数遍历
        best_n_components = get_param(train,train_X,datelist)
        
        # 3. 训练集内放入最优参数
        
        hmm = GaussianHMM(n_components = best_n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)

        data = get_signal_return(latent_states_sequence,train,datelist)

        # 4. 选择清仓的信号
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        get_signal = (cumnet.iloc[-1] < r).replace(False,np.nan).dropna().index.tolist()
        get_signal = [int(i[6:7]) for i in get_signal]                     # 转成数字形式

        # 5. 样本外计算
        test = df.iloc[i - hist_len: i + retrain_gap - 1]       # 因为有预计算所以先长的算完，再切分 （往前20天也够了）
        test['mtm1'] = test['close'].pct_change()
        test['mtm5'] = test['close'].pct_change(5)
        test['diffreturn'] = (test['high'] - test['low'])/test['low']
        test['volume_z'] = (test['volume'] - test['volume'].rolling(20).mean())/test['volume'].rolling(20).std()
        test = test.dropna()
        test = test.iloc[-(retrain_gap - 1):]
        closeidx = test['mtm1']
        test_X = np.array(test[['mtm5','diffreturn','volume_z']])
        datelist = test.index
        
        latent_states_sequence = hmm.predict(test_X)

        data = pd.DataFrame({
                'datelist': datelist,
                'mtm1': test['mtm1'],
                'state': latent_states_sequence
            }).set_index('datelist')
        
        # 6. 清仓信号叠加
        data_trade += data[data.state.isin(get_signal)].index.tolist()



    net = index_ret.to_frame('ret').iloc[hist_len:]

    net.loc[data_trade,'clear_day'] = 0           # 加上清仓信号
    net = net.fillna(1)

    for i in [0,1]:
        name = i
        state = net.clear_day == i
        idx = np.append(0, state[:-1])
        net[name] = net.ret.multiply(idx, axis=0)

    cumnet = (net.drop(['ret','clear_day'],axis = 1) + 1).cumprod()
    ax = cumnet.plot()
    cumnet
    fig=ax.get_figure()
    fig.savefig('../web/assets/threshold_adjustment1.jpg')


    r= r0-0.1
    hist_len = hist_len0
    retrain_gap = retrain_gap0
    data_trade = []
    length = df.shape[0]

    # 夏普样本加权计算
    def get_sharpe_weight(x):
        # 输入累计收益序列
        try:                                                                      # 防止进来的就没有数据
            final_ret = x.iloc[-1]
            ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
            annual_ret = final_ret ** (252/len(ret)) - 1
            std_ret = ret.std()            
            sharpe = abs(annual_ret/std_ret)                                      # 绝对值  
            sharpe = sharpe * len(ret) / len(x)                                   # 数据量修正
        except:
            sharpe = 0

        return sharpe

    def get_param(train,train_X,datelist):
        fitness_result = {}
        for n_components in range(3,8):
            try:
                hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
                latent_states_sequence = hmm.predict(train_X)
                data = pd.DataFrame({
                                        'datelist': datelist,
                                        'mtm1': train['mtm1'],
                                        'state': latent_states_sequence
                                        }).set_index('datelist')
                
                for i in range(hmm.n_components):
                    name = f'state {i}_return'
                    state = (latent_states_sequence == i)
                    idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
                    data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
                cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
                fitness = cumnet.apply(lambda x: get_sharpe_weight(x))
                fitness_result[n_components] = round(fitness.mean(),4)
            except:
                continue
        best_n_components = pd.Series(fitness_result).sort_values().index[-1]

        return best_n_components


    def get_signal_return(latent_states_sequence,train,datelist):
        data = pd.DataFrame({
            'datelist': datelist,
            'mtm1': train['mtm1'],
            'state': latent_states_sequence
        }).set_index('datelist')

        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
        
        return data

    for i in tqdm(range(hist_len,length,retrain_gap)):

        # 1. 训练集特征构建
        train = df.iloc[i-hist_len:i]
        train['mtm1'] = train['close'].pct_change()
        train['mtm5'] = train['close'].pct_change(5)
        train['diffreturn'] = (train['high'] - train['low'])/train['low']
        train['volume_z'] = (train['volume'] - train['volume'].rolling(20).mean())/train['volume'].rolling(20).std()
        train = train.dropna()
        closeidx = train['mtm1']
        train_X = np.array(train[['mtm5','diffreturn','volume_z']])
        datelist = pd.to_datetime(train.index)

        # 2. 训练集参数遍历
        best_n_components = get_param(train,train_X,datelist)
        
        # 3. 训练集内放入最优参数
        
        hmm = GaussianHMM(n_components = best_n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)

        data = get_signal_return(latent_states_sequence,train,datelist)

        # 4. 选择清仓的信号
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        get_signal = (cumnet.iloc[-1] < r).replace(False,np.nan).dropna().index.tolist()
        get_signal = [int(i[6:7]) for i in get_signal]                     # 转成数字形式

        # 5. 样本外计算
        test = df.iloc[i - hist_len: i + retrain_gap - 1]       # 因为有预计算所以先长的算完，再切分 （往前20天也够了）
        test['mtm1'] = test['close'].pct_change()
        test['mtm5'] = test['close'].pct_change(5)
        test['diffreturn'] = (test['high'] - test['low'])/test['low']
        test['volume_z'] = (test['volume'] - test['volume'].rolling(20).mean())/test['volume'].rolling(20).std()
        test = test.dropna()
        test = test.iloc[-(retrain_gap - 1):]
        closeidx = test['mtm1']
        test_X = np.array(test[['mtm5','diffreturn','volume_z']])
        datelist = test.index
        
        latent_states_sequence = hmm.predict(test_X)

        data = pd.DataFrame({
                'datelist': datelist,
                'mtm1': test['mtm1'],
                'state': latent_states_sequence
            }).set_index('datelist')
        
        # 6. 清仓信号叠加
        data_trade += data[data.state.isin(get_signal)].index.tolist()


    print(len(index_ret.to_frame('ret'))>hist_len)
    net = index_ret.to_frame('ret').iloc[hist_len:]
    net.loc[data_trade,'clear_day'] = 0           # 加上清仓信号
    net = net.fillna(1)

    for i in [0,1]:
        name = i
        state = net.clear_day == i
        idx = np.append(0, state[:-1])
        net[name] = net.ret.multiply(idx, axis=0)

    cumnet = (net.drop(['ret','clear_day'],axis = 1) + 1).cumprod()
    ax = cumnet.plot()
    cumnet
    fig=ax.get_figure()
    fig.savefig('../web/assets/thresh_adj2.jpg')


    # #### 5.3 动态阈值


    r= r0
    hist_len = hist_len0
    retrain_gap = retrain_gap0
    data_trade = []
    length = df.shape[0]

    # 夏普样本加权计算
    def get_sharpe_weight(x):
        # 输入累计收益序列
        try:                                                                      # 防止进来的就没有数据
            final_ret = x.iloc[-1]
            ret = x.pct_change().replace(0,np.nan).dropna()                       # 剔除不变化的天数
            annual_ret = final_ret ** (252/len(ret)) - 1
            std_ret = ret.std()            
            sharpe = abs(annual_ret/std_ret)                                      # 绝对值  
            sharpe = sharpe * len(ret) / len(x)                                   # 数据量修正
        except:
            sharpe = 0

        return sharpe

    def get_param(train,train_X,datelist):
        fitness_result = {}
        for n_components in range(3,8):
            try:
                hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
                latent_states_sequence = hmm.predict(train_X)
                data = pd.DataFrame({
                                        'datelist': datelist,
                                        'mtm1': train['mtm1'],
                                        'state': latent_states_sequence
                                        }).set_index('datelist')
                
                for i in range(hmm.n_components):
                    name = f'state {i}_return'
                    state = (latent_states_sequence == i)
                    idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
                    data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
                cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
                fitness = cumnet.apply(lambda x: get_sharpe_weight(x))
                fitness_result[n_components] = round(fitness.mean(),4)
            except:
                continue
        best_n_components = pd.Series(fitness_result).sort_values().index[-1]

        return best_n_components


    def get_signal_return(latent_states_sequence,train,datelist):
        data = pd.DataFrame({
            'datelist': datelist,
            'mtm1': train['mtm1'],
            'state': latent_states_sequence
        }).set_index('datelist')

        for i in range(hmm.n_components):
            name = f'state {i}_return'
            state = (latent_states_sequence == i)
            idx = np.append(0, state[:-1])                         # 当日信号 次日获取收益
            data[name] = data.mtm1.multiply(idx, axis=0)           # 收益计入对应信号中去
        
        return data

    for i in tqdm(range(hist_len,length,retrain_gap)):

        # 1. 训练集特征构建
        train = df.iloc[i-hist_len:i]
        train['mtm1'] = train['close'].pct_change()
        train['mtm5'] = train['close'].pct_change(5)
        train['diffreturn'] = (train['high'] - train['low'])/train['low']
        train['volume_z'] = (train['volume'] - train['volume'].rolling(20).mean())/train['volume'].rolling(20).std()
        train = train.dropna()
        closeidx = train['mtm1']
        train_X = np.array(train[['mtm5','diffreturn','volume_z']])
        datelist = pd.to_datetime(train.index)

        # 2. 训练集参数遍历
        best_n_components = get_param(train,train_X,datelist)
        
        # 3. 训练集内放入最优参数
        
        hmm = GaussianHMM(n_components = best_n_components, covariance_type='diag', n_iter=10000,random_state = 11).fit(train_X)
        latent_states_sequence = hmm.predict(train_X)

        data = get_signal_return(latent_states_sequence,train,datelist)

        # 4. 选择清仓的信号
        cumnet = (data.drop(['mtm1','state'],axis = 1) + 1).cumprod()
        get_signal = (cumnet.iloc[-1] < (data.mtm1 + 1).cumprod().iloc[-1]).replace(False,np.nan).dropna().index.tolist()
        get_signal = [int(i[6:7]) for i in get_signal]                     # 转成数字形式

        # 5. 样本外计算
        test = df.iloc[i - hist_len: i + retrain_gap - 1]       # 因为有预计算所以先长的算完，再切分 （往前20天也够了）
        test['mtm1'] = test['close'].pct_change()
        test['mtm5'] = test['close'].pct_change(5)
        test['diffreturn'] = (test['high'] - test['low'])/test['low']
        test['volume_z'] = (test['volume'] - test['volume'].rolling(20).mean())/test['volume'].rolling(20).std()
        test = test.dropna()
        test = test.iloc[-(retrain_gap - 1):]
        closeidx = test['mtm1']
        test_X = np.array(test[['mtm5','diffreturn','volume_z']])
        datelist = test.index
        
        latent_states_sequence = hmm.predict(test_X)

        data = pd.DataFrame({
                'datelist': datelist,
                'mtm1': test['mtm1'],
                'state': latent_states_sequence
            }).set_index('datelist')
        
        # 6. 清仓信号叠加
        data_trade += data[data.state.isin(get_signal)].index.tolist()


    net = index_ret.to_frame('ret').iloc[hist_len:]
    net.loc[data_trade,'clear_day'] = 0           # 加上清仓信号
    net = net.fillna(1)

    for i in [0,1]:
        name = i
        state = net.clear_day == i
        idx = np.append(0, state[:-1])
        net[name] = net.ret.multiply(idx, axis=0)

    cumnet = (net.drop(['ret','clear_day'],axis = 1) + 1).cumprod()
    ax = cumnet.plot()
    cumnet
    fig=ax.get_figure()
    fig.savefig('../web/assets/dynamic_threshold.jpg')


if __name__ == '__main__':
    run()