# 智能股市分析師：Python股價歷史資料自動化爬蟲與分析


## 1. **專案概述：**
這個專案的目的是透過爬蟲技術,自動化抓取所需的股價歷史資料,以協助分析師或投資者進行後續的技術或基本面分析。主要功能包括:

- 從 Yahoo Finance 上爬取指定股票的歷史股價資料,包括日期、開盤價、收盤價、成交量等。
- 計算並新增該股票的每日報酬率。
- 計算並新增移動平均線,例如 5 日、20 日均線。
- 建立交易策略,例如利用均線黃金交叉判斷買賣時機。
- 比較該股票與市場指數的報酬,檢視策略效果。
- 透過這個應用,使用者可以快速自動化下載所需的股價資料,並套用不同的技術指標,開發並測試不同的投資交易策略,以協助投資決策。它可以省去手動搜尋和匯入資料的時間,使分析工作變得更具效率。
## 2. **學習目標：**
   開發這個網路爬蟲應用程式,我主要有以下幾個學習目標:

- 掌握使用Python進行網路爬蟲的資料抓取與清洗技巧,如使用模組如pandas、matplotlib、yfinance、numpy等。
- 學習使用Python分析金融資料的方法,如計算技術指標、設計交易策略、進行回測等。
- 了解如何儲存和管理大量抓取的資料。
- 熟練使用資料視覺化模組像matplotlib,將複雜的分析結果更好的展示出來。
- 提高解決實際業務問題的能力,如根據具體的研究或分析需求來設計爬蟲程序。

透過這個計畫的學習,我提高了在Python爬蟲、數據分析和視覺化方面的技能,也訓練了獨立思考和解決問題的能力。 這些都是很有價值的能力,將幫助我更好地運用程式設計技術解決實際問題。

## 3. **技術選擇：**

**System OS:** MacOS 16.x arm64

**程式語言:** Python 

**IDE:** Google Colaboratory

**執行環境:** Jupyter Notebook

### **爬蟲框架與模組:**

- pandas:用於資料處理與分析
- pandas_datareader:提供從網路來源爬取資料的功能 
- yfinance: pandas_datareader 的擴充套件,專門用於爬取 Yahoo Finance 的資料
- datetime:處理日期時間的Python內建模組
- matplotlib:繪製圖表的模組

### **資料儲存:**

- 將爬取的股價資料儲存為CSV檔案格式,存放在Google雲端硬碟上
- 也可選擇其他資料儲存方式,如本地硬碟、數據庫等

### **程式撰寫與執行環境:**

- Google Colab：適合機器學習與資料分析專案

## 4. **功能特點：**
 
### **爬取的網站和資料:**

- 從 Yahoo Finance 上爬取台灣和美股的歷史股價資料
- 資料類型包括日期、開盤價、收盤價、最高價、最低價、成交量等

### **資料處理:**  

- 計算每日報酬率
- 計算常用的技術分析指標,如移動平均線、KD值等
- 建立交易策略並計算策略報酬,如利用均線交叉等

### **資料儲存:**

- 將爬取的原始和計算後的資料存成 CSV 檔案
- 儲存到 Google 雲端硬碟以方便管理分享

### **其他功能:**

- 可以指定任意時間範圍進行爬蟲,例如最近一年或任意起訖日期
- 支援多支股票同時爬取分析比較  

透過這些功能,可以實現自動化爬取股價資料,並進行後續的技術分析或策略回測,以協助投資決策或研究。

## 5. **應用場景：**
### **投資研究分析:**

- 投資人及分析師可以利用此應用自動爬取所需股票的歷史資料,並進行技術或基本面分析,從大量資料中尋找交易訊號。

### **算法交易策略回測:**

- 開發自動化交易策略的量化投資人,可以用此應用快速取得回測所需的歷史股價,並檢驗交易算法的效益。

### **金融資料庫建構:** 

- 金融業者可以利用此應用的爬蟲功能自動採集構建所需的股票資料庫。

### **學術研究:**

- 財金相關領域的學者可以運用此應用來爬取所需樣本進行論文研究。

### **財經新聞報導:**

- 財經媒體可以用其中的圖表等視覺化分析,來增強新聞報導的深度。

### **其他資料科學領域:**

- 該應用中的資料爬取、清理與分析技巧,也可應用在需要處理網路大數據的其他領域。
## 6. **挑戰和解決方案：**
   在開發這個網路爬蟲應用的過程中,我遇到了挑戰,以及相應的解決方案包括:

**資料儲存與管理:**

- 挑戰:爬取的海量資料如何有效存儲與管理是個挑戰。
- 解決方案:對資料進行索引。

在解決這些問題的過程中,我掌握了爬蟲、資料分析與工程化相關的技能,提高了應對挑戰的能力。
## 7. **成果展示：**
### part1 import套件：
```python=
import datetime as t#時間格式
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr_web
import yfinance as yf
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
yf.pdr_override()
```
### part2 執行爬蟲（台積電、0050股價）
```python=
symbol = '2330.TW'

start = '2018-11-27'
end = '2023-11-26'
stock_df = pdr_web.get_data_yahoo(symbol, start, end, interval='1d')

stock_df['Ret_daily'] = stock_df['Adj Close'].pct_change()
stock_df['Ret_daily%'] = stock_df['Ret_daily'] *100
stock_df['Ret_daily加1'] = stock_df['Ret_daily'] +1
stock_df['Avg_5D'] = stock_df['Adj Close'].rolling(window=5).mean()
stock_df['Avg_20D'] = stock_df['Adj Close'].rolling(window=20).mean()
stock_df['Avg_60D'] = stock_df['Adj Close'].rolling(window=60).mean()
stock_df = stock_df.drop(['Ret_daily%','Ret_daily加1'], axis=1)
print(stock_df.tail(10))
```
![截圖 2023-12-02 上午11.57.51](https://hackmd.io/_uploads/HkfpRVdHT.png)

```python=
symbol = '0050.TW'

start = '2018-11-27'
end = '2023-11-26'
stock_df = pdr_web.get_data_yahoo(symbol, start, end, interval='1d')

stock_df['Ret_daily'] = stock_df['Adj Close'].pct_change()
stock_df['Ret_daily%'] = stock_df['Ret_daily'] *100
stock_df['Ret_daily加1'] = stock_df['Ret_daily'] +1
stock_df['Avg_5D'] = stock_df['Adj Close'].rolling(window=5).mean()
stock_df['Avg_20D'] = stock_df['Adj Close'].rolling(window=20).mean()
stock_df['Avg_60D'] = stock_df['Adj Close'].rolling(window=60).mean()
stock_df = stock_df.drop(['Ret_daily%','Ret_daily加1'], axis=1)
print(stock_df.tail(10))
```
![截圖 2023-12-02 下午1.22.06](https://hackmd.io/_uploads/BkSGJSOH6.png)
### part3 畫線型圖
```python=
#直接畫線型圖，收盤價、5日均線、20日均線、60日均線
stock_df['Adj Close'].plot(figsize=(16, 8))
stock_df['Avg_5D'].plot(figsize=(16, 8), label='5_Day_Mean')
stock_df['Avg_20D']. plot(figsize=(16, 8), label='20_Day_Mean')
stock_df['Avg_60D']. plot(figsize=(16, 8), label='60_Day_Mean')

#顯示側標
plt.legend(loc='best', shadow=True, fontsize='x-large')
#顯示標題
plt.title(symbol+'_yahoo_finance')
```
![image](https://hackmd.io/_uploads/HJgv1ruBT.png)
### part4 寫入檔案
```python=
plt.title(symbol+'_yahoo_finance')
from google.colab import drive
drive_name='/content/google_drive'
folder_name = "/content/google_drive/My Drive/Python課程檔案寫入區/"
ﬁle_mode='w'
file_name = '2330.TW_2023Nov.csv'
 # 檔案名稱為:  股票代碼_日期.csv 檔  
drive.mount(drive_name) # 掛載到google drive
with open(folder_name + file_name, ﬁle_mode) as f: 
  stock_df.to_csv(f)
drive.flush_and_unmount() # 卸載 
```
### part5 讀取台積電、0050股價存檔
```python=
#讀取兩個dataframe
import pandas as pd
from google.colab import drive
drive_name='/content/google_drive'
drive.mount(drive_name)
folder_name = "/content/google_drive/My Drive/Python課程檔案寫入區/"

file_name = '2330.TW_2023Nov.csv' #每次執行前，請先確認檔案名稱
df_read = pd.read_csv(folder_name + file_name, encoding='utf-8')

file_name = '0050.TW_2023Nov.csv' #每次執行前，請先確認檔案名稱
df_read_0050 = pd.read_csv(folder_name + file_name, encoding='utf-8')

drive.flush_and_unmount() #卸載Google drive
print(df_read.tail()) 
print(df_read_0050.tail()) 
```
![截圖 2023-12-02 下午2.05.10](https://hackmd.io/_uploads/SJwmYHOr6.png)
### part6 (資料合併)
```python=
#Merge 測試
## 先整理0050檔案，重新命名
df_read_0050 = df_read_0050.rename(columns={'Ret_daily': 'Ret_daily_0050'})

## 創造一個只包含 Date 和 Return 的 dataframe，以便後續合併 Dataframe
df_0050_ret = df_read_0050[['Date', 'Ret_daily_0050']]
print(df_0050_ret.tail(5))

## 將0050的Return 合併至各股的 dataframe (範例為上述讀取的 2330.TW_2022-11-28)
## 合併後，存入新的dataframe，命名為stock_df
stock_df = pd.merge(df_read , df_0050_ret, on='Date')
print(stock_df.tail(10))
```
![截圖 2023-12-02 下午2.07.42](https://hackmd.io/_uploads/BJmptr_rp.png)
### part7 (複利計算)
```python=
## 累積複利計算
import numpy as np
stock_df['Ret加1'] = stock_df['Ret_daily'] +1 #計算複利時可使用
stock_df['Ret_5D_backward'] = stock_df['Ret加1'].rolling(window=5).apply(np.prod)

#往後算五日的複利報酬
stock_df['Ret_5D_forward'] = stock_df['Ret加1'].shift(-5).rolling(window=5).apply(np.prod) 

## 計算 個股(2330) 的未來20日累積複利報酬
stock_df['Ret_20D_forward'] = stock_df['Ret加1'].shift(-20).rolling(window=20).apply(np.prod)
#計算 台灣五十(0050) 的未來20日累積複利報酬
stock_df['Ret_0050加1'] = stock_df['Ret_daily_0050'] +1 #計算複利時可使用
stock_df['Ret_20D_forward_0050'] = stock_df['Ret_0050加1'].shift(-20).rolling(window=20).apply(np.prod)

print(stock_df.tail(10))
```
![截圖 2023-12-02 下午2.09.01](https://hackmd.io/_uploads/ryt-5H_ra.png)
### part8 (設定黃金交叉日)
```python=

#計算移動平均
stock_df['Avg_5D'] = stock_df['Adj Close'].rolling(window=5).mean()
stock_df['Avg_10D'] = stock_df['Adj Close'].rolling(window=10).mean()

# 定義 5日移動平均 (MA5) > 10日移動平均 (MA10)
stock_df['MA5_above_MA10'] = np.where(stock_df['Avg_5D']>stock_df['Avg_10D'], 1, 0)

# 定義黃金交叉: 5日平均 從小於(等於)變為大於 10日移動平均 
# shift(1) 取得Row的前一筆資料
stock_df['Golden_MA5_MA10'] = np.where( stock_df['MA5_above_MA10'] > stock_df['MA5_above_MA10'].shift(1) , 1, 0)

# 檢查用
print(stock_df[['Date', 'Avg_5D', 'Avg_10D', 'MA5_above_MA10', 'Golden_MA5_MA10']].head(20))
```
![截圖 2023-12-02 下午2.10.50](https://hackmd.io/_uploads/SyDOcB_Sa.png)

### part9 (樣本整理與統計描述)
```python=
#只保留 黃金交叉日的資料，以便比較
#創立一個新的df，只保留想要用的資料
df_Gold_MA5_MA10 = stock_df[['Date', 'Golden_MA5_MA10', 'Ret_20D_forward', 'Ret_20D_forward_0050']]

#只保留 黃金交叉日的資料，以便比較
df_Gold_MA5_MA10 = df_Gold_MA5_MA10.loc[df_Gold_MA5_MA10['Golden_MA5_MA10'] == 1]

print(df_Gold_MA5_MA10)
print(df_Gold_MA5_MA10.describe())

# 去除包含遺失值(NaN)的row
df_no_NaN = df_Gold_MA5_MA10.dropna(subset=['Ret_20D_forward', 'Ret_20D_forward_0050'])

display(df_no_NaN.describe())
```
![截圖 2023-12-02 下午2.11.49](https://hackmd.io/_uploads/H1On5SurT.png)
### part10 結果
此策略下，個股的20日報酬之平均值(Mean)，比0050的20日報酬之平均值(Mean)還要高，故從結果可以得知黃金交叉策略可能有效。
![截圖 2023-12-02 下午2.25.42-2](https://hackmd.io/_uploads/HJBUl8drT.jpg)


## 8. **未來發展：**

### **建立資料庫:**

- 將爬蟲到的資料匯入mysql資料庫，以便更好的管理。

### **新增更多源站:**

- 目前主要爬取 Yahoo Finance,未來可擴展至其他財經網站,提供更豐富的資料來源。

### **雲端部署:**

- 在雲服務平台上自動部署和擴展該應用,提供更穩定的服務。



