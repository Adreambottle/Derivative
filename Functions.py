import pandas as pd
import numpy as np
import tushare as ts
import datetime
import math
from scipy.special import comb

from math import log, sqrt, exp

class download_data():

    def __init__(self, SC, sta_date, end_date):
        self.SC = SC
        self.sta_date = sta_date
        self.end_date = end_date
        self.token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
        self.pro = ts.pro_api(self.token)

    def get_stock_code(self):
        """
        获取股票代码
        """
        return self.SC

    def set_stock_code(self, SC):
        """
        设置股票代码
        """
        self.SC = SC

    def get_stock_data(self):
        """
        获取股票交易数据
        """
        df = self.pro.daily(ts_code=self.SC,
                            start_date = self.sta_date,
                            end_date=self.end_date)
        df.index = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.sort_index(inplace=True)
        return df

    def get_fund_data(self):
        """
        获取基金交易数据
        """
        df = self.pro.fund_daily(ts_code=self.SC,
                            start_date = self.sta_date,
                            end_date=self.end_date)
        df.index = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.sort_index(inplace=True)
        return df

    def get_dividend(self):
        """
        获取股票股息
        """
        # pro.dividend(ts_code=SC, fields='ts_code,div_proc,stk_div,record_date,ex_date')
        df = self.pro.dividend(ts_code=self.SC, fields='ts_code,div_proc,stk_div,record_date,ex_date')
        return df

    def get_fund_dividend(self):
        """
        获取基金股息
        """
        # pro.dividend(ts_code=SC, fields='ts_code,div_proc,stk_div,record_date,ex_date')
        df = self.pro.fund_div(ts_code=self.SC, fields='ts_code,div_proc,stk_div,record_date,ex_date')
        return df


    def get_r(self, type = "shibor"):
        """
        得到无风险利率
        :param type: 采用的是上海银行拆借利率shibor
        :return: 返回的是年利率，一个DataFrame
        """
        if type == "libor":
            df = self.pro.libor(curr_type='USD',
                                start_date=self.sta_date,
                                end_date=self.end_date)
        if type == "shibor":
            df = self.pro.shibor(start_date=self.sta_date,
                                 end_date=self.end_date)
        if type == "hibor":
            df = self.pro.hibor(curr_type='RMB',
                                start_date=self.sta_date,
                                end_date=self.end_date)

        df.index = pd.to_datetime(df["date"], format="%Y%m%d")
        df.sort_index(inplace=True)
        return df["1y"]


def get_volatility(df, method = "percentage"):
    """
    得到一直股票的历史波动率
    :param df: 股票的历史数据，最好采用每天的收盘价
    :param method: 有两种方法，默认的是百分比法
    :return:
    """
    # df = close
    length = df.shape[0]
    Xi_list = []
    for i in range(0, length-1):
        # i = 0

        # 百分比价格变动法
        if method == "percentage":
            Xi = (df.iloc[i+1] - df.iloc[i]) / df.iloc[i]

        # 对数价格变动法
        elif method == "logarithm":
            Xi = math.log(df.iloc[i + 1]) - math.log(df.iloc[i])

        Xi_list.append(Xi)
    Xi_list = np.array(Xi_list)
    X_bar = Xi_list.mean()
    sigma = math.sqrt(sum([(x - X_bar)**2 for x in Xi_list]))
    return sigma

def find_hist_sigma(SC):
    Project = download_data(SC, '20100101', '20190101')
    close_data = Project.get_stock_data()["close"]
    sigma = get_volatility(close_data)
    return sigma


def get_option_by_s(ST, X, alpha = 0.3):
    """
    这个期权与股票S的关系，分为三种情况
    :param ST: 股票在t=T的时候的价格
    :param X: Strike Price
    :param alpha: 定价因子alpha，在(0, 1)之间
    :return: 返回的是期权的价格
    """
    if alpha * ST >= ST - X >= 0:
        option = ST - X
    if ST - X > alpha * ST:
        option = alpha * ST
    if ST - X < 0:
        option = 0
    return option


def Price_by_BT_2(r, sigma, delta, X, S0, h, alpha = 0.3):
    """
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: 股息dividend
    :param X: Strike Price
    :param S0: 股票在t=0的价格
    :param h: 一次二叉树的时间
    :param alpha: 奇异期权的定价因子
    :return: 返回的是一个dictionary
    """

    u = math.exp((r-delta) * h + sigma*math.sqrt(h))
    d = math.exp((r-delta) * h - sigma*math.sqrt(h))
    p = (math.exp((r-delta) * h) - d)/(u-d)
    q = 1 - p
    uS = u * S0
    dS = d * S0
    uOp = get_option_by_s(uS, X, alpha)
    dOp = get_option_by_s(dS, X, alpha)
    sanjiao = math.exp(-delta*h) * (uOp - dOp)/(S0*(u-d))
    B = math.exp(-r*h) * (u * dOp - d * uOp)/(u-d)
    output_dic = {"Call_Up":uOp,
                  "Call_down":dOp,
                  "Stock_Up":uS,
                  "Stock_Down":dS,
                  "Up_factor":u,
                  "Down_factor":d,
                  "p_star":p,
                  "delta":sanjiao,
                  "B":B}
    return output_dic



def Build_Tree_E(n, r, sigma, delta, X, S0, h, alpha):
    """
    欧式期权的二叉树
    :param n: 进行n次回归
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: dividend的
    :param X: Strike Price
    :param S0_P: 之前股票的价格
    :param h: 每次二叉树的间隔
    :param alpha: 这个是奇异期权的定价中的变量alpha
    :return: 返回是一个嵌套list
    """

    stock_matrix = np.zeros((n+1)*(n+1)).reshape(((n+1), (n+1)))
    option_matrix = np.zeros((n+1)*(n+1)).reshape(((n+1), (n+1)))
    stock_matrix[0, 0] = S0
    for j in range(0, n):
        for i in range(0, j+1):
            S0_P = stock_matrix[i][j]
            if j != i:
                stock_matrix[i, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
            if j == i:
                stock_matrix[i, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                stock_matrix[i+1, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Down"]
    p = Price_by_BT_2(r, sigma, delta, X, S0, h, alpha)["p_star"]
    q = 1 - p
    for i in range(0, n+1):
        option_matrix[i, n] = get_option_by_s(stock_matrix[i, n], X, alpha)
    for j in range(n-1, -1, -1):
        for i in range(0, j+1):
            # print(i, j)
            option_matrix[i, j] = p * option_matrix[i, j+1] + q * option_matrix[i+1, j+1]
    return option_matrix

def Build_Tree_A(n, r, sigma, delta, X, S0, h, alpha):
    """
    美式期权的二叉树
    :param n: 进行n次回归
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: dividend的
    :param X: Strike Price
    :param S0_P: 之前股票的价格
    :param h: 每次二叉树的间隔
    :param alpha: 这个是奇异期权的定价中的变量alpha
    :return: 返回是一个嵌套list
    """

    stock_matrix = np.zeros((n+1)*(n+1)).reshape(((n+1), (n+1)))
    option_matrix = np.zeros((n+1)*(n+1)).reshape(((n+1), (n+1)))
    stock_matrix[0, 0] = S0
    for j in range(0, n):
        for i in range(0, j+1):
            S0_P = stock_matrix[i][j]
            if j != i:
                stock_matrix[i, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
            if j == i:
                stock_matrix[i, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                stock_matrix[i+1, j+1] = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Down"]

    p = Price_by_BT_2(r, sigma, delta, X, S0, h, alpha)["p_star"]
    q = 1 - p
    for i in range(0, n+1):
        option_matrix[i, n] = get_option_by_s(stock_matrix[i, n], X, alpha)
    for j in range(n-1, -1, -1):
        for i in range(0, j+1):
            # print(i, j)
            price_1 = p * option_matrix[i, j+1] + q * option_matrix[i+1, j+1]
            price_2 = get_option_by_s(stock_matrix[i, j], X, alpha)
            option_matrix[i, j] = max(price_1, price_2)

    return option_matrix


# r = 0.035
# n = 10
# delta = 0.02
# sigma = 1.2
# h = 1/365
# X = 120
# S0 = 100
# alpha = 0.3
# a = Build_Tree_A(n, r, sigma, delta, X, S0, h, alpha)
# b = Build_Tree_E(n, r, sigma, delta, X, S0, h, alpha)


def BS_Formular(X, alpha, sigma, delta, r, ST, T, t):
    K = X/(1 - alpha)
    d1 = (log(ST/X) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
    d2 = d1 - sigma*sqrt(T-t)
    d3 = (log(ST/K) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
    d4 = d3 - sigma*sqrt(T-t)



def main():
    SC = "510050.SH"
    print("The stock code is: ", SC)
    sta_date = "20200901"
    end_date = "20201001"
    print("The start date and the end date are: ", sta_date, " ", end_date)

    # 下载数据
    project = download_data(SC=SC, sta_date=sta_date, end_date=end_date)
    project.get_stock_code()

    # 数据用data存储的
    data = project.get_fund_data()
    data.to_excel("/Users/meron/Desktop/data.xlsx")

    # 获取股息数据，这个只能人定
    delta_df = project.get_dividend()
    delta_df
    delta = 0.2/100

    # 获取无风险利率
    r = project.get_r().mean()/100

    # 获取历史波动率
    sigma = find_hist_sigma(SC)

    # 设定 Strike Price
    X = 1

    # 设定alpha
    alpha = 0.5

    # 期权的时间是什么
    n = data.shape[0]
    h = 1/365

    # 设定股票在t=0的价格
    S0 = data.iloc[0]

a = 1 + \
    3
    X = [14, 15, 16]
    n = [10, 100, 200]
    T = np.array([7, 30, 365])/365
    alpha = [0.1, 0.5, 1]

    price_dict = []
    for i in X:
        for j in n:
            for l in alpha:
                for m in T:
                    European_Option_Price = Build_Tree_A(j, r, sigma, delta, i, S0, m/j, l)[0, 0]
                    American_Option_Price = Build_Tree_E(j, r, sigma, delta, i, S0, m/j, l)[0, 0]
                    option = {"K":i,
                              "n":j,
                              "T":m,
                              "alpha":l,
                              "EP":European_Option_Price,
                              "AP":American_Option_Price}
                    price_dict.append(option)

    df = pd.DataFrame(price_dict)
    # 建立欧式期权和美式期权的BT
    European_Option_BT = Build_Tree_A(n, r, sigma, delta, X, S0, h, alpha)
    American_Option_BT = Build_Tree_E(n, r, sigma, delta, X, S0, h, alpha)

    # 获取价格
    European_Option_Price = European_Option_BT[0, 0]
    American_Option_Price = American_Option_BT[0, 0]


