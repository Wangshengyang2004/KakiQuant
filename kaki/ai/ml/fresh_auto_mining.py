from kaki.datafeed.reader.MongoDataReader import DownloadData
from kaki.factor.ta.ta import cal_ta_factors
import pandas as pd
import os
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh import extract_features
import logging
from tsfresh import select_features
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def preprocess_crypto(data):
    data = data.drop(columns=['instId', 'bar']).sort_index().fillna(method='ffill').dropna()
    data['code'] = "000001"
    # 插入日期列
    data.insert(0, 'date', data.index)
    # 将日期从datetime格式转换为str格式
    data['date'] = data['timestamp']
    data.drop(columns=['timestamp','confirm'], inplace=True)
    data = data.reset_index(drop=True)
    return cal_ta_factors(data)
     
def main(data):
    data_roll = roll_time_series(data, column_id='code', column_sort='date', max_timeshift=20, min_timeshift=5).drop(columns=['code'])  
    gg = data_roll.groupby('id').agg({'date':['count', min, max]}) 
    data_feat = extract_features(data_roll, column_id='id', column_sort='date', n_jobs=63)
    # 对单独标的而言，将日期作为index
    data_feat.index = [v[1] for v in data_feat.index]
    # 将原始因子加入因子矩阵当中
    data_feat = pd.merge(data_feat, data.set_index('date', drop=True).drop(columns=['code']), 
                        how='left', left_index=True, right_index=True)

    # 给数据打标签
    data_feat['pct'] = data_feat['close'].shift(-1) / data_feat['close'] - 1.0
    data_feat['rise'] = data_feat['pct'].apply(lambda x: 1 if x>0 else 0)
    data_feat = data_feat.dropna(subset=['pct'])
    logging.info(f"data_feat's shape is {data_feat.shape}")
    logging.info([ i for i in data_feat.columns])
    # 划分训练集和测试集
    num_train = round(len(data_feat)*0.8)
    data_train = data_feat.iloc[:num_train, :]
    y_train = data_feat.iloc[:num_train, :]['rise']
    data_test = data_feat.iloc[num_train:, :]
    y_test = data_feat.iloc[num_train:, :]['rise']

    # 特征选择
    data_train0 = select_features(data_train.drop(columns=['pct','rise']).dropna(axis=1, how='any'), y_train)
    select_columns = list(data_train0.columns) + ['pct','rise']
    data_train = data_train[select_columns]
    data_test = data_test[select_columns]

    logging.info(f"training set shape: {data_train.shape}")
    
    # 转化为numpy的ndarray数组格式
    X_train = data_train.drop(columns=['pct','rise']).values
    X_test = data_test.drop(columns=['pct','rise']).values

    # 对数据进行标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型
    # classifier = DecisionTreeClassifier(max_depth=5)
    classifier = SVC(C=1.0, kernel='rbf')
    classifier.fit(X_train, y_train)
    
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    data_train['pred'] = y_train_pred
    data_test['pred'] = y_test_pred
    accuracy_train = 100 * data_train[data_train.rise==data_train.pred].shape[0] / data_train.shape[0]
    accuracy_test = 100 * data_test[data_test.rise==data_test.pred].shape[0] / data_test.shape[0]
    logging.info('训练集预测准确率：%.2f%%' %accuracy_train)
    logging.info('测试集预测准确率：%.2f%%' %accuracy_test)


    #策略日收益率
    data_test['strategy_pct'] = data_test.apply(lambda x: x.pct if x.pred>0 else -x.pct, axis=1)

    #策略和沪深300的净值
    data_test['strategy'] = (1.0 + data_test['strategy_pct']).cumprod()
    data_test['baseline'] = (1.0 + data_test['pct']).cumprod()

    # 粗略计算年化收益率
    annual_return = 100 * (pow(data_test['strategy'].iloc[-1], 250/data_test.shape[0]) - 1.0)
    print('SVM 择时策略的年化收益率：%.2f%%' %annual_return)

    #将索引从字符串转换为日期格式，方便展示
    data_test.index = pd.to_datetime(data_test.index)
    ax = data_test[['strategy','hs300']].plot(figsize=(16,9), color=['SteelBlue','Red'],
                                            title='SVM 择时策略净值')
    plt.show()

if "__name__" == "__main__":
    reader = DownloadData(target="crypto")
    data = reader.download(symbol="BTC-USDT-SWAP", bar= "1m", fields="full")
    data = preprocess_crypto(data)
    main(data)
