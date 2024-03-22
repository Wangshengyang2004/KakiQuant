from tsfresh.utilities.dataframe_functions import roll_time_series
import pandas as pd
from kaki.factor.ta.ta import cal_ta_factors
from tsfresh import select_features, extract_features
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import sys, psutil
from kaki.kkplot.plot import plot_corrhmap
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
CPU_CORES = psutil.cpu_count()

class TsfreshMining:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    
    def ta_add(self) -> None:
        self.df = cal_ta_factors(self.df).dropna().reset_index()
        

    def preprocess(self, max_timeshift: int = 60, min_timeshift: int =5):
        self.ta_add()
        logging.debug(self.df.tail(10))
        logging.debug(self.df.columns)
        logging.debug(self.df.dtypes)
        # self.df.drop(columns=['bar','confirm'], inplace=True)
        self.df_roll = roll_time_series(self.df, 
                                   column_id='instId', column_sort='timestamp', 
                                   max_timeshift=max_timeshift, min_timeshift=min_timeshift
                                   ).drop(columns=['instId']).reset_index(drop=True)
        # self.df_roll.to_csv('df_roll.csv')
        logging.debug(self.df.shape)
    
    def extract_feats(self):
        self.df_roll.drop(columns=['bar','confirm'], inplace=True)
        self.df_features = extract_features(self.df_roll, column_id='id', column_sort='timestamp', n_jobs=CPU_CORES)
        self.df_features.index = [v[1] for v in self.df_features.index]
        self.df_features = pd.merge(self.df_features, self.df.set_index('timestamp', drop=True).drop(columns=['instId']), 
                     how='left', left_index=True, right_index=True)
        # 给数据打标签
        self.df_features['pct'] = self.df_features['close'].shift(-1) / self.df_features['close'] - 1.0
        self.df_features['rise'] = self.df_features['pct'].apply(lambda x: 1 if x>0 else 0)
        self.df_features = self.df_features.dropna(subset=['pct'])
        plot_corrhmap(self.df_features)
        logging.debug(self.df_features.shape)
    
    def train(self):
        # 划分训练集和测试集
        num_train = round(len(self.df_features)*0.8)
        data_train = self.df_features.iloc[:num_train, :]
        y_train = self.df_features.iloc[:num_train, :]['rise']
        self.df_test = self.df_features.iloc[num_train:, :]
        y_test = self.df_features.iloc[num_train:, :]['rise']

        # 特征选择
        data_train0 = select_features(data_train.drop(columns=['pct','rise']).dropna(axis=1, how='any'), y_train)
        select_columns = list(data_train0.columns) + ['pct','rise']
        data_train = data_train[select_columns]
        self.df_test = self.df_test[select_columns]
        # 转化为numpy的ndarray数组格式
        X_train = data_train.drop(columns=['pct','rise']).values
        X_test = self.df_test.drop(columns=['pct','rise']).values

        # 对数据进行标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练模型
        classifier = SVC(C=1.0, kernel='rbf')
        classifier.fit(X_train, y_train)

        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        data_train['pred'] = y_train_pred
        self.df_test['pred'] = y_test_pred
        accuracy_train = 100 * data_train[data_train.rise==data_train.pred].shape[0] / data_train.shape[0]
        accuracy_test = 100 * self.df_test[self.df_test.rise==self.df_test.pred].shape[0] / self.df_test.shape[0]
        logging.info('训练集预测准确率：%.2f%%' %accuracy_train)
        logging.info('测试集预测准确率：%.2f%%' %accuracy_test)
        self.df_test = self.df_test
    
    def plot(self) -> None:
        #策略日收益率
        self.df_test['strategy_pct'] = self.df_test.apply(lambda x: x.pct if x.pred>0 else -x.pct, axis=1)

        #策略和沪深300的净值
        self.df_test['strategy'] = (1.0 + self.df_test['strategy_pct']).cumprod()
        self.df_test['hs300'] = (1.0 + self.df_test['pct']).cumprod()

        # 粗略计算年化收益率
        annual_return = 100 * (pow(self.df_test['strategy'].iloc[-1], 250/self.df_test.shape[0]) - 1.0)
        print('SVM 沪深300指数择时策略的年化收益率：%.2f%%' %annual_return)

        #将索引从字符串转换为日期格式，方便展示
        self.df_test.index = pd.to_datetime(self.df_test.index)
        ax = self.df_test[['strategy','hs300']].plot(figsize=(16,9), color=['SteelBlue','Red'],
                                                title='SVM 沪深300指数择时策略净值')
        plt.show()


    def main(self):
        self.preprocess()
        self.extract_feats()
        self.train()
        self.plot()

if __name__ == "__main__":
    from kaki.kkdatac.crypto import get_crypto_price
    import time
    t = time.time()
    df = get_crypto_price(instId='BTC-USDT-SWAP', bar='1D')
    print(f'cost:{time.time() - t:.4f}s to read db')
    ts = TsfreshMining(df)
    ts.preprocess()
    ts.main()