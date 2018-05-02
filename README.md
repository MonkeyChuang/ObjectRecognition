# 科學計算軟體 期末報告 影像辨識

## 簡述
### 組員
數學系 莊沛聖
左邊才是我，右邊是助教(΄◉◞౪◟◉‵)
~~(圖源已技術性死亡)~~
[](https://i.imgur.com/LvqWVX9.jpg)

### 使用語言
Python
### 動機與描述

曾經在某個介紹Neural Network的影片裡看到有人可以用攝影機和一些程式，就可以讓程式去學習他拍攝到的影像，並且去判斷接下來拍到的東西是什麼。覺得挺好玩的，所以就做一個來玩玩看。

具體來說就是要寫一個程式，並且能夠：
1. 捕捉並儲存攝影機的內容(Training Dataset)
2. 如果某段影像被設定為某個「物體」，那要幫那段影像多加上標記(Class Label)
3. 選定一個Classifier，並用上述的Training Dataset與Class Label來訓練它
4. 讓訓練過後的Classifier來預測接下來拍攝到的畫面是什麼東西
5. 搭配GUI來執行上述的操作

## 手法

### 影像捕捉
「把影像從攝影機抓下來」有很多種不同的方法，像是可以用筆電內建的攝影鏡頭、外接的視訊鏡頭、手機App [IP WebCam](https://play.google.com/store/apps/details?id=com.pas.webcam)(Android)或[WebOfCam](https://play.google.com/store/apps/details?id=com.webofcam&hl=zh_TW)(iOS與Android皆支援)依據不同的方法，也有不同的捕捉方式，這邊就粗略地依序說明：

#### 內建/外接鏡頭
需要使用`OpenCV (ver=3.0 or 3.1)`這個模組來執行的影像的操作。簡單的操作與說明如下：
```python
import cv2    
#連接上影像裝置，參數代表了該裝置的index，內建鏡頭一般都是0
cap = cv2.VideoCapture(0)    
#讀取鏡頭的影像，並array的型態存到image當中。不知道ret代表的意思...
ret,image = cap.read()       
```
#### 使用手機App
電腦與手機均需要連接上相同的WIFI。而程式則需使用`urllib`套件讀取串流到網路中的影像內容。更詳細的介紹在[這裡](https://hackmd.io/JwEwTAxmAMBsDsBaAHAMwIwkQFlQU3RTG20XWj2WHlFQGY7kg===?view)。

### 為特定影像加上標記(Class Label)
畢竟要由使用者只會告訴程式「眼前這個畫面是什麼什麼」，而在訓練classifier時用的Class Label習慣上都是以「數字」來表示，所以我們需要一個能從「對應的數字編號」找到「物體的名稱」的資料結構。因此我採用的方式就是用python內建的dictionary來完成這個任務。
就像是這樣：
```python
dict = {
    1:'蘋果'
    2:'西瓜'
    3:'助教'
    }
#如果想要知道key 1對應的value為何時就輸入
dict[1]    
#Output:'蘋果'
```

### 影像儲存
儲存的方式很簡單，只要用`numpy`的函數`vstack()`就好。比較困擾的是「同一個物體」需要用多少幀(Frame)的影像來代表？為了不要降低程式的效能，我暫且就以30幀來儲存和特定物體相關的影像。

### Classifier的選擇與使用
機器學習裡面有一大堆的可以使用的classifier，畢竟我瞭解的也不是特別深入，所以姑且先以SVM來作為我的classifier。在python裡面，`scikit-learn(ver=0.18)`套件提供了很多種機器學習可以使用上的東西，當然包含了SVM。大致上的使用方式如下：
```python
from sklearn.svm import SVC

#初始化classifier，kernel function使用linear，penalty parameter=50.0
classifier = SVC(kernel='linear',C=50.0)

#使用Training dataset(trainX)與Class Label(labelY)來訓練classifier
classifier.fit(trainX,labelY)

#讓訓練完畢的classifier去猜測Unknown data(image)是哪種類別。
predY = classifier.predict(image)
```

### GUI
我就用`tkinter(ver=8.15.18)`來負責GUI的建構。
由於使用`OpenCV`或`urllib`讀取到的影像是以`array`的格式儲存，然而在`tkinter`當中的影像處理方式卻是以它獨有的`PhotoImage`的物件在操作，所以使用了`PIL`套件中的`Image`與`ImageTk`模組來把`array`型態的圖片轉為`PhotoImage`的物件


## 結果

### 初始畫面
![](https://i.imgur.com/YdWAYxK.png)
- 左側是系統回覆區，主要在回報程式運行間發生的狀況，例如：提供的IP錯誤導致無法連接上手機鏡頭，或是完成了training data的蒐集，等等情形。
- 中間上方則是影像呈現區塊，值得注意的是不管影像大小為何，都會壓縮成720x480的大小再顯示出來。
- 中間下方提供了三種影像來源的選擇，第二種「Built-in Camera」就是指筆電內建鏡頭，如果本身沒有內建鏡頭，則是指外接的鏡頭。如果選擇了第一種或第三種，要記得在右側的輸入格打上用來瀏覽影像的IP位置。點選下方的Start即可開始，而且按鍵上的文字會變為「Connecting」並回報連線進度到系統回報區。連線成功後，此按鍵的文字會顯示為「Pause」，點擊它則會終止鏡頭的連接。

- 右上區塊是用來記錄已被蒐集完成的Training Data的標記(Class Label)與其對應的物體名稱。如果想要蒐集一段新的Training Data，就在這區塊下方的輸入格打上你想要蒐集的影像中的物體名稱，再按下「Collect training data」按鈕來開始蒐集。總計蒐集30幀影像，左側系統回報區會回報蒐集的進度。在蒐集的過程中==建議不要讓鏡頭靜止不動，如果能在過程中適度的改變拍攝角度會讓學習效果更好==

- 右下區塊則是兩個最主要的功能：訓練Classifier、預測未知影像的名稱。在蒐集了至少兩種的Training data後，就可以點選「Train Classifier」來訓練我們的Classifier。訓練進度會回報在系統回報區。在尚未訓練過Classifier前，下方的「Predict」按鈕是無法被點擊的。一旦訓練好了以後，就可以點擊「Predict」按鈕來預測鏡頭畫面中的未知影像為何，預測的結果會顯示在此按鈕右方。

:::info
1. 在初始化的階段裡，除了「Start」以外的按鍵皆無效。
2. 用來接收「影像中的物體名稱」的輸入格不能接受中文輸入，原因不明。
::: 

### 其餘畫面

#### 連接上鏡頭後
![](https://i.imgur.com/Y1e5l1z.png)
#### 蒐集Training Data
![](https://i.imgur.com/VXerlER.png)
#### 訓練Classifier
![](https://i.imgur.com/nzoXHNl.png)
#### 預測未知影像
==預測結果會輸出在右下角「Predict」按鍵的右方。==
預測正確
![](https://i.imgur.com/11acgly.png)

預測正確
![](https://i.imgur.com/JFGgnOz.jpg)

預測失敗，他是Rudin
![](https://i.imgur.com/h1IFFuI.jpg)

## 問題與討論
### 影像捕捉——不同方法間的優缺點
#### 內建/外接鏡頭
- 使用內建/外接鏡頭的缺點在於攝影機不好攜帶，因為它可能是內建在電腦裡，或是用USB連接到電腦，如果想要拍到更多的畫面，就得要帶著電腦一起移動。不過他們絕對性的優勢在於影像的傳輸速率相對於另一個方法而言非常快，因為他們是以有線的方式重送資訊到電腦，而非藉由無線網路傳送。

#### 手機App
- 用手機軟體的好處當然就是: 不用另外買鏡頭，用手機鏡頭就可以、有WIFI(這WIFI甚至不需要連接上Internet，這些軟體只是藉由WIFI這種無線的管道把影像傳送到WIFI連接的區網當中)就可以使用。但另一方面，這些軟體卻只能在連接WIFI的情況下運作，如果用的是行動數據(3G、4G)就不行。而且訊號的強弱也會影像傳輸的流暢度。
- 畢竟開發這些手機軟體也是由他人開發的，所以我們沒辦法更改軟體，好讓我們直接從手機介面去執行這個程式上的許多操作，像是如果你要程式蒐集Training data，那你還是要回到電腦上操作。

### 尚未解決的Bug與沒加上去的功能
- 在選擇內建/外接鏡頭當作影像來源時，在點選「Pause」鍵中斷連線後會有段Error訊息傳到stderr當中
- 如果Training Data的種類沒有超過兩種(含)前是不能去訓練Classifier的，但我忘了把這個提示加進去。

## 程式執行說明
### 預備
1. 不建議在Spyder或任何可能也是由Tkinter設計的的環境下執行，有可能會Crash。我是直接在終端機輸入
    ```bash
    $ python project.py
    ```
1. 記得使用Python 3以上的版本
1. 如果你有安裝過Anaconda，記得再多安裝以下兩種套件
    - OpenCV (ver=3.0 or 3.1)
    - scikit-learn (ver=0.18)

### 函數功能說明
#### main
都是用來初始化gui使用到的widget，或是宣告一些變數。
#### connect(*args)
按下「Start」按鍵後會呼叫此函數。他會依據目前的狀態來判斷是要連接(Start)還是終止連接鏡頭(Pause)，並根據這兩種不同的狀態來該改功能配置。如果是在「Start」的狀態下，connect函數會在結束之前呼叫`connectURL()`或`connectCam()`。
#### connectCam()
連接上內建/外接鏡頭，並更改某些功能的配置。結束前呼叫`stream()`。
#### connectURL()
連接上使用者提供的URL。這個函數只會在使用者選擇用IP WebCam或WebOfCam來當作影像輸入源時才會被呼叫。結束前呼叫`stream()`。
#### stream()
從已經連接上的影片串流源頭，去一幀一幀地抓出影像並輸出到影像顯示區塊中。
#### fetchFrame(*args)
呼叫之後會更改某些變數，使得`stream()`函數在讀取影像之後把該影像以Training Data的身份被存起來。
#### doneFetch(*args)
附屬於`fetchFrame()`的一個函數，本身就只是用來檢查系統是否已經蒐集完30幀的影像。
#### setTrainClassifier(*args):
按下「Train Classifier」按鍵後會呼叫此函數。他會把已經收集好的Training Dataset與他們對應的標記(Class Label)拿來訓練classifier。
#### setImgClassify(*args)
按下「Predict」按鍵後會呼叫此函數。它被呼叫後會使得`stream()`把讀取的未知影像傳給classifier，讓其預測這個未知影像是屬於哪種標記(Class Label)。

## 參考資料
- Python 3.6.0 documentation: 
https://docs.python.org/3/
- Numpy Documentation:
https://docs.scipy.org/doc/
- OpenCV 3.0.0-dev documentation:
http://docs.opencv.org/3.0-beta/index.html
- Tk Documentation:
http://www.tcl.tk/man/tcl8.5/TkCmd/contents.htm
- Pillow(PIL fork) 4.0.0 documentation:
http://www.books.com.tw/products/0010728558
- 《Python機器學習》:
http://www.books.com.tw/products/0010728558


## 後記

### 進度＆狀況

#### 進度
- [x] 可以用OpenCV讀到即時的影像了: D
- [x] 讓程式要求使用者輸入"這段影像裡的東西是什麼呢？"
    > 這就是用來當作Training data的Label的
- [x] 做個GUI來插入即時的影像、開始Train、開始Test之類的功能
- [x] SVM是好物，目前想不到可以替代他的Classifier。Neural Network暫不考慮，能力尚淺。

#### 狀況
1.  大概要花多少的frame(或時間)去收集某一組資料呢？
1.  急缺視訊攝影機，不喜歡筆電的前置鏡頭，常常會被自己嚇到！
1. 有嘗試過用IP webcam + 自己的手機鏡頭來回傳即時的video stream。不過要不是網路太慢一直timeout，就是我不知道怎麼把mjpeg當中的圖片抓出來(網路上很多方法，但都失敗了，大概是版本不通？或是我太笨？)
    > [time=Sat, Dec 24, 2016]赫然發現[網路上的版本](http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera)只有在python2底下才會動，看起來是跟urllib這個package有關係@@
1. 如果要做到即時的辨識，那我的classifier就不能等收集完一組資料後才全部(包含先前收集過的資料)train過一遍，這樣做效率會隨時間降低。有沒有一個classifier是可以迭代地(iteratively)去train他呢？
    > 我目前只知道有所謂Active learning的東西...



## 老師、助教回覆區ㄏㄏ
>[name=Yu Hsun Lee][time=Fri, Dec 23, 2016 11:20 PM]不要再八七把自己的hachmd刪掉了。