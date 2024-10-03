#VS Code 安裝 Jupyter Extension 後可以在程式碼區塊上按 Shift+Enter 執行
#在 VS Code 右邊的 Terminal 視窗可以看到執行結果
#這支程式碼是使用 spaceship_titanic.py 練出來的模型(spaceship_titanic_lr_20241002.pkl) 驗證測試資料並產生 Kaggle 參賽要求的 csv 格式檔案
import joblib
import pandas as pd
import math

#------讀取模型------
model_pretrained = joblib.load("spaceship_titanic_lr-20241002.pkl")
df_test = pd.read_csv("data/test.csv")

df_test.head()
df_test.info()
df_test.isnull().sum()

#移除不會使用的欄位
df_test.drop(['HomePlanet','Cabin','Destination','Name'],axis=1,inplace=True)

#------清理資料------

#填補 Age 欄位的空值, 以中位數填補
df_test['Age'].value_counts()
age_mean = math.floor(df_test['Age'].value_counts().mean())
df_test['Age'].fillna(
    age_mean,
    inplace=True
)

#填補 CryoSleep 欄位的空值
df_test['CryoSleep'].value_counts()
df_test['CryoSleep'].value_counts().idxmax()
df_test['CryoSleep'].fillna(
    df_test['CryoSleep'].value_counts().idxmax(),
    inplace=True
)

#填補 VIP 欄位的空值
df_test['VIP'].value_counts()
df_test['VIP'].value_counts().idxmax()
df_test['VIP'].fillna(
    df_test['VIP'].value_counts().idxmax(),
    inplace=True
)

#將Cabin 欄位拆開為 deck,num,side 三個欄位
# df_test['Deck'] = df_test['Cabin'].str[0]
# df_test['Num'] = df_test['Cabin'].str[1:].str.extract('(\d+)').astype(float)
# df_test['Side'] = df_test['Cabin'].str[-1]

#填補 RoomService 欄位的空值, 以中位數填補
df_test['RoomService'].value_counts()
roomservice_median = df_test['RoomService'].value_counts().median()
df_test['RoomService'].fillna(
    roomservice_median,
    inplace=True
)

#填補 VRDeck 欄位的空值, 以中位數填補
df_test['VRDeck'].value_counts()
vrdeck_median = df_test['VRDeck'].value_counts().median()
df_test['VRDeck'].fillna(
    vrdeck_median,
    inplace=True
)

#填補 Spa 欄位的空值, 以中位數填補
df_test['Spa'].value_counts()
spa_median = df_test['Spa'].value_counts().median()
df_test['Spa'].fillna(
    spa_median,
    inplace=True
)

#填補 ShoppingMall 欄位的空值, 以中位數填補
df_test['ShoppingMall'].value_counts()
shoppingmall_median = df_test['ShoppingMall'].value_counts().median()
df_test['ShoppingMall'].fillna(
    shoppingmall_median,
    inplace=True
)

#填補 FoodCourt 欄位的空值, 以中位數填補
df_test['FoodCourt'].value_counts()
foodcourt_median = df_test['FoodCourt'].value_counts().median()
df_test['FoodCourt'].fillna(
    foodcourt_median,
    inplace=True
)

df_test.isnull().sum()
df_test.info()

#將 CryoSleep, VIP 進行轉換數值化
# df_test = pd.get_dummies(
#     data=df_test, 
#     dtype=int, 
#     columns=['CryoSleep','VIP']
# )

# df_test.drop(['CryoSleep_False','VIP_False'],axis=1,inplace=True)

#把 PassengerId 欄位存到另一個Series 變數中
# passengerId = df_test['PassengerId']

#刪除 PassengerId 欄位
# df_test.drop(['PassengerId'],axis=1,inplace=True)

#------預測結果------
#驗證 model_pretrained 預測結果
predicions2 = model_pretrained.predict(df_test)

#------產生 Kaggle 參賽要求的 csv 格式檔案------
#準備submission的檔案
for_submissionDf = pd.DataFrame({
    "PassengerId": df_test['PassengerId'],
    "Transported": predicions2
})

#產出 Kaggle 參賽要求的 csv 格式檔案 
for_submissionDf.to_csv("data/for_submission_20241002.csv",index=False)