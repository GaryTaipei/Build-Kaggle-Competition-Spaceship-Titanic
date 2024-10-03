## Build-Kaggle-Competition-Spaceship-Titanic

Git Clone 程式
1.找個資料夾進入command 視窗然後下指令程式抓下來,例如 C:\virtualenv>git clone https://github.com/GaryTaipei/Build-Kaggle-Competition-Spaceship-Titanic.git  

2.Git Clone 會產生一個 Build-Kaggle-Competition-Spaceship-Titanic 資料夾  

3.針對 Build-Kaggle-Competition-Spaceship-Titanic 資料夾建立Python 虛擬目錄 C:\virtualenv>python -m venv Build-Kaggle-Competition-Spaceship-Titanic  

4.開啟 Visual Studio Code 選檔案>開啟資料夾，選擇 Competition-Spaceship-Titanic  

5.將 VS Code 的解譯器設定為剛建立的虛擬目錄從 檢視>命令區 >Pyhton: 選取解譯器(Select Interpreter) > 選輸入解譯器路徑... >按下 "尋找",   

6.然後從彈出的瀏覽視窗選擇 C:\virtualenv\Build-Kaggle-Competition-Spaceship-Titanic\Scripts\Python.exe 這樣就可以將解譯器設定為虛擬環境的 Python 執行程式  


安裝需要的 Extension  
1.安裝 Jupyter 套件，在spaceship_titanic.py 及 for_submission.py 分段執行，在 python 程式從上而下，將要執行的程式選取後按下 shift + enter 就可以在右邊的執行視窗顯示執行結果。  

2.在 VS Code 終端機輸入 pip install -r requirements.txt 安裝所需要的延伸套件。  

3.spaceship_titanic.py 程式用來讀取訓練資料(data/train.csv)訓練並匯出模型檔。  

4.for_submission.py 用來載入模型並驗證測試資料(data/test.csv), 最後依照Kaggle網站規定的格式(data/sample_submission.csv)產出csv檔案。  

