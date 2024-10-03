#這支程式在 jupyter notebook 上分段執行
#VS Code 安裝 Jupyter Extension 後可以在程式碼區塊上按 Shift+Enter 執行
#在 VS Code 右邊的 Terminal 視窗可以看到執行結果
#這支程式用來訓練模型，並將模型儲存成檔案 spaceship_titanic_lr_20241002.pkl

import pandas as pd #pip install pandas
import numpy as np #pip install numpy
import matplotlib.pyplot as plt #pip install matplotlib
import seaborn as sns #pip install seaborn
import math

#-----讀取資料-----
df =pd.read_csv('data/train.csv')
df.head()
df.info()

#移除不會使用的欄位
#axis=0是index,axis=1是欄位, inplace=True是直接在原本的df上修改
df.drop(['HomePlanet','Cabin','Destination','Name'],axis=1,inplace=True)

#繪製 Survived 跟其他欄位的兩兩關係圖
sns.pairplot(df[['Transported','RoomService']],dropna=True)
sns.pairplot(df[['Transported','FoodCourt']],dropna=True)
sns.pairplot(df[['Transported','Spa']],dropna=True)
sns.pairplot(df[['Transported','VRDeck']],dropna=True)
sns.pairplot(df[['Transported','Age']],dropna=True)
sns.pairplot(df[['Transported','VIP']],dropna=True)
sns.pairplot(df[['Transported','CryoSleep']],dropna=True)

#將 8693 筆資料依照 Transported 分組，並計算平均值
df.groupby('Transported').mean(numeric_only=True) #numeric_only=True 只顯示數值型態的欄位
df.groupby('Transported')['VIP'].value_counts() #成功或是不成功被運送的旅客中 VIP與非VIP的人數
df['VIP'].value_counts() #VIP與非VIP的人數
df['CryoSleep'].value_counts()
df['Cabin'].value_counts()
df['HomePlanet'].value_counts()

df.isnull().sum() #計算某欄位為空值
df.isnull().sum().sort_values(ascending=False) #計算某欄位為空值，並排序
df.isnull().sum() > len(df)/2 #計算某欄位為空值，並判斷是否超過一半

df.groupby('Transported')['Age'].median() #以Transported分組，計算各性別的年齡中位數
# df.groupby('Transported')['Age'].median().plot(kind='bar') #畫成長條圖

#------清理資料------
#填補 Age 欄位的空值，以中位數填補
df['Age'].value_counts()
age_median = df['Age'].value_counts().median()
df['Age'].fillna(
    age_median,
    inplace=True
)

#填補 CryoSleep 欄位的空值為 頻率最高發生值:'Flase'
df['CryoSleep'].value_counts()
df['CryoSleep'].value_counts().idxmax()
df['CryoSleep'].fillna(
    df['CryoSleep'].value_counts().idxmax(),
    inplace=True
)

#填補 VIP 欄位的空值為 頻率最高發生值:'False'
df['VIP'].value_counts()
df['VIP'].value_counts().idxmax()
df['VIP'].fillna(
    df['VIP'].value_counts().idxmax(),
    inplace=True
)

#填補 RoomService 欄位的空值為中位數
df['RoomService'].value_counts()
roomservice_median = df['RoomService'].value_counts().median()
df['RoomService'].fillna(
    roomservice_median,
    inplace=True
)

#填補 FoodCourt 欄位的空值為中位數
df['FoodCourt'].value_counts()
foodcourt_median = df['FoodCourt'].value_counts().median()
df['FoodCourt'].fillna(
    foodcourt_median,
    inplace=True
)

#填補 ShoppingMall 欄位的空值為中位數
df['ShoppingMall'].value_counts()
shoppingmall_median = df['ShoppingMall'].value_counts().median()
df['ShoppingMall'].fillna(
    shoppingmall_median,
    inplace=True
)

#填補 Spa 欄位的空值為中位數
df['Spa'].value_counts()
spa_median = df['Spa'].value_counts().median()
df['Spa'].fillna(
    spa_median,
    inplace=True
)

#填補 VRDeck 欄位的空值為中位數
df['VRDeck'].value_counts()
vrdeck_median = df['VRDeck'].value_counts().median()
df['VRDeck'].fillna(
    vrdeck_median,
    inplace=True
)

#將Cabin 欄位拆開為 deck,num,side 三個欄位
# df['Deck'] = df['Cabin'].str[0]
# df['Num'] = df['Cabin'].str[1:].str.extract('(\d+)').astype(float)
# df['Side'] = df['Cabin'].str[-1]

#補填 Deck 欄位的空值為 頻率最高發生值:'C'
# df['Deck'].value_counts()
# df['Deck'].value_counts().idxmax()
# df['Deck'].fillna(
#     df['Deck'].value_counts().idxmax(),
#     inplace=True
# )
# #補填 Num 欄位的空值為 平均值
# df['Num'].value_counts()
# num_mean = math.floor(df['Num'].value_counts().mean())
# df['Num'].fillna(
#     num_mean,
#     inplace=True
# )

# #補填 Side 欄位的空值為 頻率最高發生值:'S'
# df['Side'].value_counts()
# df['Side'].value_counts().idxmax()
# df['Side'].fillna(
#     df['Side'].value_counts().idxmax(),
#     inplace=True
# )

df.isnull().sum()
df.head()
df.info()

#將 Transported 進行轉換數值化
# df = pd.get_dummies(
#     data=df, 
#     dtype=int, 
#     columns=['Transported']
# )

# df.drop(['Transported_False'],axis=1,inplace=True)
# df.rename(columns={'Transported_True':'Transported'},inplace=True)

# CryoSleep_False,  VIP_False, Transported_False 刪除，留下 CryoSleep_True, VIP_True, Transported_True
# df.drop(['CryoSleep_False','VIP_False','Transported_False'],axis=1,inplace=True)

#把 CryoSleep_True, VIP_True, Transported_True 改名為 CryoSleep, VIP, Transported
# df.rename(columns={'CryoSleep_True':'CryoSleep','VIP_True':'VIP','Transported_True':'Transported'},inplace=True)

#----訓練模型----

df.corr() #把欄位數值倆倆分析相關性

X=df.drop(['Transported'],axis=1) 
y=df['Transported'] #目標值

from sklearn.model_selection import train_test_split #pip install scikit-learn
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=67)

# using Logistic regression model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000) #初始化
lr.fit(X_train,y_train)

#-----使用訓練好的模型進行預測-----
#匯出訓練後的模型
import joblib
joblib.dump(lr,'spaceship_titanic_lr-20241002.pkl', compress=3)

predictions = lr.predict(X_test)

#Evaluate
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,predictions) #準確率 0.7822085889570553
recall_score(y_test,predictions) #召回率 0.7972560975609756
precision_score(y_test,predictions) #精確率 0.7759643916913946

pd.DataFrame(
    confusion_matrix(y_test,predictions),
    columns=['Predict not Transported', 'Predict Transported'],
    index=['True not Transported','True Transported']
    )

#Precision 準確率 = 模型預測爲𝒀𝒆𝒔且實際上爲𝒀𝒆𝒔 / 模型預測爲𝒀𝒆𝒔的個數
#Recall 召回率 = 實際上爲𝒀𝒆𝒔而模型也預測爲𝒀𝒆𝒔 / 實際上爲𝒀𝒆𝒔的所有個數
#Accuracy 精準率 = 模型預測爲𝒀𝒆𝒔且實際上爲𝒀𝒆𝒔$+模型預測爲𝑵𝒐且實際上爲𝑵𝒐 / 所有預測的個數
#F1 Score = 2*(Precision*Recall)/(Precision+Recall) 
precision_score = 1046/(302 +1046) = 0.7759643916913946
recall_score = 1046/ (266 + 1046) = 0.7972560975609756
F1_score = 2*(precision_score*recall_score)/(precision_score+recall_score) = 0.7864661654135338