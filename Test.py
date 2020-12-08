import pandas as pd
import numpy as np
import tushare as ts
import math
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt

SC = "510050.SH"
print("The stock code is: ",SC)
sta_date = "20200901"
end_date = "20201001"
print("The start date and the end date are: ",sta_date," ",end_date)

# 下载数据
project = download_data(SC=SC, sta_date=sta_date, end_date=end_date)
project.get_stock_code()