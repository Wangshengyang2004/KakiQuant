from scipy.stats import rankdata

import scipy as sp
import numpy as np
import pandas as pd

class KDCJ_003:
    
    def __init__(self,open,high,low,close,prev_close,volume,amount,avg_price):

        # volume	high	num_trades	prev_close	limit_down	limit_up	total_turnover	close	open	low

        self.open_price = open
        self.close      = high
        self.low        = low
        self.high       = close
        self.prev_close = prev_close
        self.volume     = volume
        self.amount     = amount
        self.avg_price  = avg_price


        #########################################################################

    def func_rank(self, na):
        return rankdata(na)[-1]/rankdata(na).max()
    
    def func_decaylinear(self, na):
        n = len(na)
        decay_weights = np.arange(1,n+1,1) 
        decay_weights = decay_weights / decay_weights.sum()

        return (na * decay_weights).sum()
    
    def func_highday(self, na):
        return len(na) - na.values.argmax()

    def func_lowday(self, na):
        return len(na) - na.values.argmin()
    
    #############################################################################
    
    def alpha_001(self):
        data1 = self.volume.diff(periods=1).rank(axis=1,pct=True)
        data2 = ((self.close - self.open_price)/self.open_price).rank(axis=1,pct=True)
        alpha = -data1.iloc[-6:,:].corrwith(data2.iloc[-6:,:]).dropna()
        return alpha
    
    def alpha_002(self):
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        result=((self.close-self.low)-(self.high-self.close))/((self.high-self.low)).diff()
        m=result.iloc[-1,:].dropna() 
        alpha=m[(m<np.inf)&(m>-np.inf)]      
        return alpha.dropna()


    ################################################################
    def alpha_003(self):  
        delay1 = self.close.shift()   
        condtion1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)

        part2 = (self.close-np.minimum(delay1[condition2],self.low[condition2])).iloc[-6:,:]
        part3 = (self.close-np.maximum(delay1[condition3],self.low[condition3])).iloc[-6:,:] 

        result=part2.fillna(0)+part3.fillna(0)
        alpha=result.sum()
        return alpha.dropna()