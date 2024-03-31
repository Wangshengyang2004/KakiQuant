import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os
from ta1 import *
import datetime
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
end = datetime.datetime.now()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-3,3))
import gm.api as gm
gm.set_token('')
fields = ['open','high', 'low', 'close', 'volume']

# 获取指数历史数据
index = gm.history(symbol='SHSE.000905', frequency='1d', start_time='2019-01-01', end_time='2024-03-21', adjust=gm.ADJUST_PREV, df=True)
index.index = index['bob']  # 将日期列设置为索引
indexre = np.array(index['close'])  # 提取指数收盘价数据
indexre = (indexre[1:] - indexre[0:len(indexre)-1]) / indexre[0:len(indexre)-1]  # 计算指数收益率
indexre = pd.DataFrame(indexre)
indexre.index = index.index[1:]

# 获取中证500指数成分股
zz500 = gm.stk_get_index_constituents(index='SHSE.000905')
zz500 = zz500['symbol'].to_list()
#zz500=np.load('zz500.npy')
data = []
for i in range(0, len(zz500)):
    # 获取成分股历史数据
    klines0 = gm.history(symbol=zz500[i], frequency='1d', start_time='2019-01-01', end_time='2024-03-21', adjust=gm.ADJUST_PREV, df=True)
    klines0 = klines0[klines0['eob'] > '2019-01-01']
    klines0.index = klines0['bob']
    klines0 = pd.DataFrame(klines0, columns=fields)
    klines0 = klines0.T
    z1 = []
    z2 = []
    for i1 in klines0.index:
        z1.append(zz500[i])
        z2.append(i1)
    klines0.index = pd.MultiIndex.from_arrays([z1, z2])
    klines0 = klines0.T
    data.append(klines0)

# 合并成分股数据
data = pd.concat([data, data][0], axis=1)
data = data.dropna(axis=1, how='any')
data = data.ffill()
code=data.T.index
zz500=[]
for i in code:
    zz500.append(i[0])
zz500=list(set(zz500))
price = []
for i in zz500:
    price.append(data[i]['open'])

price = pd.DataFrame(price)
price.index = zz500
price1 = price.T

data = data[zz500]
data1 = pd.DataFrame(data)
zz500 = price1.T.index.tolist()


# 加载并转换为列表
fuc = np.load('fuc1.npy').tolist()

fuc1 = []
function = []
fc = []

# 遍历fuc列表
for i in fuc:
    fuc1.append([i[0], int(i[1])])  # 将第二个元素转换为整数并添加到fuc1列表中
    function.append(i[0])  # 将第一个元素添加到function列表中
    fc.append(eval(i[0]))  # 将第一个元素作为表达式求值，并添加到fc列表中

fuc = fuc1  # 更新fuc为fuc1的值

# 加载并截取yz变量
yz=pd.read_excel('zz500.xls')
re = []
# 遍历 zz500 的长度范围
for i in range(0, len(zz500)):
    data2 = data1[zz500[i]]
    open = data2['open'].ffill()  # 对 open 列进行向前填充
    close = data2['close'].ffill()  # 对 close 列进行向前填充
    low = data2['low'].ffill()  # 对 low 列进行向前填充
    high = data2['high'].ffill()  # 对 high 列进行向前填充
    volume = data2['volume'].ffill()  # 对 volume 列进行向前填充
    re1 = (open.values[1:] - open.values[0:len(open) - 1]) / open.values[0:len(open) - 1]  # 计算 re1 值
    re.append(re1)  # 将 re1 添加到 re 列表中

dt = data1.index  # 获取 data1 的索引
re = pd.DataFrame(re)  # 将 re 列表转换为 DataFrame 对象
cd = []  # 创建一个空列表

# 嵌套循环生成cd列表
# 嵌套循环生成 cd 列表
for ii in range(0, len(dt) - 1):
    for ii1 in zz500:
        cd.append([dt[ii], ii1])
cd = pd.DataFrame(cd)  # 将 cd 列表转换为 DataFrame 对象
re1 = []
# 遍历 re.T 的长度范围
for ii in range(0, len(re.T)):
    re1.append(re.iloc[:, ii])  # 提取 re 的每列，并添加到 re1 列表中
re1 = pd.concat([re1, re1][0])  # 将 re1 列表进行连接
re1.index = pd.MultiIndex.from_arrays([cd[0].values, cd[1].values], names=['datetime', 'instrument'])  # 为 re1 设置多重索引，使用 cd 列表的值作为索引名称
yinzi1 = []
# 遍历 yz 的长度范围
def st1(yz,data1,zz500,function,fc,ii):
    warnings.filterwarnings("ignore")
    print(ii)
    for i in range(0, len(function)):
        globals()[function[i]] = fc[i]
    try:
        yinzi1 = []
        # 遍历 zz500 的长度范围
        for i in range(0, len(zz500)):
            data2 = data1[zz500[i]]
            open = data2['open'].ffill()
            close = data2['close'].ffill()
            low = data2['low'].ffill()
            high = data2['high'].ffill()
            volume = data2['volume'].ffill()
            z = (eval(yz[0][ii]))  # 使用 eval() 方法计算 yz 的元素值
            yinzi1.append(z)  # 将计算结果添加到 yinzi 列表中
        yinzi = sc.fit_transform(yinzi1)
        yinzi = pd.DataFrame(yinzi)  # 将 yinzi 列表转换为 DataFrame 对象
        yinzi = yinzi.fillna(0)  # 将 NaN 值填充为 0
        yinzi = yinzi.replace(np.inf, 0)  # 将无穷大的值替换为 0
        yinzi.index = zz500  # 设置 yinzi 的索引为 zz500
        yinzi = yinzi.T  # 转置 yinzi
        return yinzi
        #yinzi1.append(yinzi)  # 将 yinzi 添加到 yinzi1 列表中
    except:
        pass

yinzi1=Parallel(n_jobs=os.cpu_count()-2)(delayed(st1)(yz,data1,zz500,function,fc,ii) for ii in range(0, len(yz)))
yinzi1 = [x for x in yinzi1 if x is not None]

yinzi2 = []

# 遍历 yinzi1 的长度范围
for i in range(0, len(yinzi1)):
    try:
        z = yinzi1[i]
        yinzi3 = []

        # 遍历 dt 的长度范围
        for ii in range(0, len(dt) - 1):
            yinzi3.append(z.iloc[ii, :])  # 提取 z 的每行数据，并添加到 yinzi3 列表中

        yinzi3 = pd.concat([yinzi3, yinzi3][0])  # 将 yinzi3 列表进行连接
        yinzi2.append(yinzi3.values)  # 将 yinzi3 的值添加到 yinzi2 列表中
    except:
        pass

yinzi2 = pd.DataFrame(yinzi2)  # 将 yinzi2 列表转换为 DataFrame 对象

yinzi2 = yinzi2.T  # 转置 yinzi2
yinzi2.index = pd.MultiIndex.from_arrays([cd[0].values, cd[1].values], names=['datetime', 'instrument'])  # 设置 yinzi2 的多重索引
yinzi2['re']=re1
yinzi2 = yinzi2.dropna(axis=0, how='any')  # 删除包含 NaN 值的行
yinzi2 = yinzi2.ffill()  # 对 yinzi3 进行向前填充
yinzi2 = yinzi2.replace([np.inf, -np.inf], np.nan)  # 将无穷大和无穷小的值替换为 NaN
yinzi2 = yinzi2.dropna(axis=0, how='any')  # 删除包含 NaN 值的行
re1 = yinzi2['re']  # 提取 yinzi3 的第一列作为 re1
yinzi2 = yinzi2.drop(columns='re')  # 删除 yinzi3 的第一列，得到 yinzi2
yinzi2 = tf.keras.utils.normalize(yinzi2.T)  # 对 yinzi2 进行归一化处理
yinzi2 = yinzi2.T  # 再次转置 yinzi2

x_train = yinzi2['2019-01-01':'2022-5-01']  # 从 yinzi2 中提取训练集特征数据，时间范围为 '2016-08-07' 到 '2019-1-15'
y_train = re1['2019-01-01':'2022-5-01']  # 从 re1 中提取训练集目标数据，时间范围为 '2016-08-07' 到 '2019-1-15'
x_valid = yinzi2['2022-5-01':'2024-03-10']  # 从 yinzi2 中提取验证集特征数据，时间范围为 '2019-1-16' 到 '2022-05-01'
y_valid = re1['2022-5-01':'2024-03-10']  # 从 re1 中提取验证集目标数据，时间范围为 '2019-1-16' 到 '2022-05-01'
x_test = yinzi2['2024-03-11':'2024-03-21']  # 从 yinzi2 中提取测试集特征数据，时间范围为 '2022-05-01' 到 '2023-05-01'
y_test = re1['2024-03-11':'2024-03-21']  # 从 re1 中提取测试集目标数据，时间范围为 '2022-05-01' 到 '2023-05-01'

x_train = np.array(x_train).reshape(len(x_train), 1, int(len(x_train.T)))  # 将训练集特征数据转换为 numpy 数组，并调整形状
x_valid = np.array(x_valid).reshape(len(x_valid), 1, int(len(x_valid.T)))  # 将验证集特征数据转换为 numpy 数组，并调整形状
x_test = np.array(x_test).reshape(len(x_test), 1, int(len(x_test.T)))  # 将测试集特征数据转换为 numpy 数组，并调整形状

tf.keras.backend.clear_session()  # 清除当前的 TensorFlow 会话
callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=False)]  # 设置回调函数，用于早停策略
from tcn import TCN
model = tf.keras.models.Sequential()
model.add(TCN(input_shape=x_train.shape[-2:]))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=20, batch_size=20000, verbose=1, validation_data=(x_valid, y_valid),callbacks=callbacks)
model.save('tcn.keras')