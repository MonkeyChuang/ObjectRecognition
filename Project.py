##############################################
# READ this before moving on:
#   https://hackmd.io/BwU2ENwYwdgFgLQEZQDMFwEwFYBsCBOAZinSQAZNw5gZUYK4g===?view
##############################################
from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from sklearn.svm import SVC
import cv2
import urllib.request,urllib.error
import numpy as np

def connectCam():
    root.stream = cv2.VideoCapture(0)

    startbutton.state(['!disabled'])
    startpause.set('Pause')

    urlentry.state(['disabled','!focus'])
    urlentry2.state(['disabled','!focus'])

    labelbutton.state(['!disabled'])
    trainbutton.state(['!disabled'])

    sysreport.configure(state='normal')
    sysreport.insert('end -1 char','>連線成功\n')
    sysreport.configure(state='disabled')
    streaming() 

def connectURL():
    try:
        if sourcetype.get() == 'ipcam':
            root.stream = urllib.request.urlopen('http://'+urlvar.get()+'/video?.mjpg')
        else:
            root.stream = urllib.request.urlopen('http://'+urlvar2.get())
            
    except urllib.error.URLError:
        switch.set(not switch.get())

        startbutton.state(['!disabled'])
        startpause.set('start(ENTER)')

        source1.state(['!disabled'])
        source2.state(['!disabled'])
        source3.state(['!disabled'])

        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>連線異常，要不是提供的IP不對，就是網路不穩，再試一次\n')
        sysreport.configure(state='disabled')

    else:
        startbutton.state(['!disabled'])
        startpause.set('Pause')
        urlentry.state(['disabled','!focus'])
        urlentry2.state(['disabled','!focus'])

        labelbutton.state(['!disabled'])
        trainbutton.state(['!disabled'])

        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>連線成功\n')
        sysreport.configure(state='disabled')
        streaming()

def connect(*args):
    switch.set(not switch.get())
    if switch.get():
        startpause.set('Connecting...')
        startbutton.state(['disabled'])

        source1.state(['disabled'])
        source2.state(['disabled'])
        source3.state(['disabled'])

        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>正在連接相機...\n')
        sysreport.configure(state='disabled')

        if sourcetype.get() == 'ipcam' or sourcetype.get() == 'webofcam':
            root.after(1,connectURL)
        elif sourcetype.get() == 'buildin':
            root.after(1,connectCam)
            
    else:
        if sourcetype.get() == 'buildin':
            root.stream.release()
        startpause.set('Start')

        source1.state(['!disabled'])
        source2.state(['!disabled'])
        source3.state(['!disabled'])

        urlentry.state(['!disabled'])
        urlentry2.state(['!disabled'])

        labelbutton.state(['disabled'])
        trainbutton.state(['disabled'])
        predbutton.state(['disabled'])

        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>連線中斷\n')
        sysreport.configure(state='disabled')

def fetchFrame(*args):
    if labelvar.get() != '':
        root.labelcount += 1
        root.label_dict[root.labelcount] = labelvar.get()
        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>蒐集Training data中...\n'+'(Class Label:'+labelvar.get()+')\n')
        sysreport.configure(state='disabled') 
        labelbutton_info.set('Collecting...')
        labelbutton.state(['disabled'])
        fetchframe.set(30)
    else:
        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>Class Label欄位不能是空的\n')
        sysreport.configure(state='disabled') 
        labelentry.focus()
def doneFetch(*args):
    if fetchframe.get() == 0:
        labelbutton_info.set('Collect training data')
        labelbutton.state(['!disabled'])
        studylist.configure(state='normal')
        studylist.insert('end -1 char','\n'+str(root.labelcount)+': '+labelvar.get())
        studylist.configure(state='disabled')
        sysreport.configure(state='normal')
        sysreport.insert('end -1 char','>蒐集完成\n')
        sysreport.configure(state='disabled')

def setTrainClassifier(*args):
    sysreport.configure(state='normal')
    sysreport.insert('end -1 char','>開始訓練...\n')
    sysreport.configure(state='disabled') 

    root.classifier.fit(root.trainX,root.labelY)
    
    sysreport.configure(state='normal')
    sysreport.insert('end -1 char','>訓練完成.\n')
    sysreport.configure(state='disabled') 

    predbutton.state(['!disabled'])

def setImgClassify(*args):
    root.img_classify = not root.img_classify
def streaming(*args):
    if sourcetype.get() == 'ipcam' or sourcetype.get() == 'webofcam':
        get_pic = False
        buffer = bytes()
        while not get_pic:
            buffer += root.stream.read(2000)
            first = buffer.find(b'\xff\xd8')
            last = buffer.find(b'\xff\xd9')
            if last < first:
                last = buffer.find(b'\xff\xd9',last+1)
            if first!=-1 and last!=-1:
                get_pic = True
                img = buffer[first:last+2]
                buffer = buffer[last+2:]
                img = cv2.imdecode(np.frombuffer(img,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                h,w=img.shape
                if np.ndim(img) == 3:
                    b,g,r = cv2.split(img)
                    img = cv2.merge((r,g,b))
                if fetchframe.get():
                    root.trainX = np.vstack((root.trainX,img.reshape((1,h*w)))) if root.trainX.size != 0 else img.reshape((1,h*w))
                    root.labelY = np.append(root.labelY,root.labelcount)
                    print('count=%d'%(fetchframe.get()))
                    fetchframe.set(fetchframe.get()-1)
                if root.img_classify:
                    setImgClassify()
                    Y_pred = root.classifier.predict(img.reshape((1,h*w)))
                    predAnsvar.set('類別:'+root.label_dict[Y_pred[0]])
                img = Image.fromarray(img)
                img = img.resize((720,480))
                imgtk = ImageTk.PhotoImage(image=img)
                window.imgtk = imgtk
                window.configure(image=imgtk)
                if switch.get():
                    window.after(5,streaming)
    elif sourcetype.get() == 'buildin':
        ret,img = root.stream.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h,w=img.shape
        if np.ndim(img) == 3:
            b,g,r = cv2.split(img)
            img = cv2.merge((r,g,b))
        if fetchframe.get():
            root.trainX = np.vstack((root.trainX,img.reshape((1,h*w)))) if root.trainX.size != 0 else img.reshape((1,h*w))
            root.labelY = np.append(root.labelY,root.labelcount)
            print('count=%d'%(fetchframe.get()))
            fetchframe.set(fetchframe.get()-1)
        if root.img_classify:
            setImgClassify()
            Y_pred = root.classifier.predict(img.reshape((1,h*w)))
            predAnsvar.set('類別:'+root.label_dict[Y_pred[0]])

        img = Image.fromarray(img)
        img = img.resize((720,480))
        imgtk = ImageTk.PhotoImage(image=img)
        window.imgtk = imgtk
        window.configure(image=imgtk)
        if switch.get():
            window.after(5,streaming)
# 宣告root
root = Tk()
root.bind('<Escape>',lambda e:root.quit())
root.bind('<Return>',connect)
root.title('IP webcam')
root.configure(bg='#c0c0c0')
root.stream = None
root.classifier = SVC(kernel='linear',random_state=0,C=50.0)
root.trainX = np.array([])
root.labelY = np.array([],dtype=np.uint8)
root.labelcount = 0
root.label_dict={}
root.img_classify = False

fetchframe = IntVar(value=0)
switch = BooleanVar(value=False)

# 宣告window，用來顯示影像的視窗。bg是初次使用時的預設背景
bg = Image.open('test.png')
bg = bg.resize((720,480))
bg = ImageTk.PhotoImage(bg)
window = ttk.Label(root,image=bg,borderwidth=1,relief='solid')
window.grid(row=1,column=1,padx=5,pady=5)


# 用來選擇影像輸入源的widget
sourceframe = ttk.Frame(root,height=50,width=500,borderwidth=1,relief='sunken')
urlvar = StringVar()
urlvar2 = StringVar()
sourcetype = StringVar(value='ipcam')
startpause = StringVar(value='start')

source1 = ttk.Radiobutton(sourceframe,text='IP WebCam',variable=sourcetype,value='ipcam')
source2 = ttk.Radiobutton(sourceframe,text='Built-in Camera',variable=sourcetype,value='buildin')
source3 = ttk.Radiobutton(sourceframe,text='WebOfCam',variable=sourcetype,value='webofcam')

urllab_f = ttk.Label(sourceframe,text='http://')
urllab_r = ttk.Label(sourceframe,text='/video?.mjpg')
urlentry = ttk.Entry(sourceframe,textvariable=urlvar,width=20)

urllab_f2 = ttk.Label(sourceframe,text='http://')
urlentry2 = ttk.Entry(sourceframe,textvariable=urlvar2,width=30)

startbutton = ttk.Button(sourceframe,textvariable=startpause,command=connect)
urlentry.state(['focus'])
urlentry.insert(0,'192.168.100.29:8080')
urlentry2.insert(0,'192.168.100.29:8080/video.jpeg?sessionId=1484485279957')
source1.grid(row=1,column=1,sticky=W,padx=5)
source2.grid(row=2,column=1,sticky=W,padx=5)
source3.grid(row=3,column=1,sticky=W,padx=5)
urllab_f.grid(row=1,column=2,sticky=E)
urlentry.grid(row=1,column=3)
urllab_r.grid(row=1,column=4,sticky=W)
urllab_f2.grid(row=3,column=2,sticky=E)
urlentry2.grid(row=3,column=3)
startbutton.grid(row=4,column=2,padx=5,pady=5)
sourceframe.grid(row=2,column=1)

# 用來定義、蒐集training data的widget，studylist用來顯示已蒐集過的training data的資訊
labelframe = ttk.Frame(root,borderwidth=1,relief='sunken')
labelvar = StringVar()
labelbutton_info = StringVar(value='Collect training data')
labellab = ttk.Label(labelframe,text='輸入影像中的物體名稱：') 
labelentry = ttk.Entry(labelframe,textvariable=labelvar,width=15)
labelbutton = ttk.Button(labelframe,textvariable=labelbutton_info,command=fetchFrame)
labelbutton.state(['disabled']) 
labellab.grid(row=2,padx=2,pady=2,sticky=W)
labelentry.grid(row=3)
labelbutton.grid(row=4)

studylist = Text(labelframe,width=20,height=20,wrap='char',borderwidth=1,relief='sunken')
studylist.configure(state='normal')
studylist.insert('1.0','Study List:')
studylist.configure(state='disabled')
studylist.grid(row=1,padx=2,pady=2)
labelframe.grid(row=1,column=2)

fetchframe.trace('w',doneFetch)

# 操作classifier相關的動作，像是訓練辨識器、預估未知影像的類別
clfframe = ttk.Frame(root,borderwidth=1,relief='sunken')
predAnsvar = StringVar(value='類別：??')
clflabel = ttk.Label(clfframe,text='Classifier的操作')
trainbutton = ttk.Button(clfframe,text='Train Classifier',command=setTrainClassifier)
predbutton = ttk.Button(clfframe,text='Predict',command=setImgClassify)
predAnslabel = ttk.Label(clfframe,textvariable=predAnsvar)
trainbutton.state(['disabled'])
predbutton.state(['disabled'])
clflabel.grid(row=1,column=1,padx=5,sticky=W)
trainbutton.grid(row=2,column=1,sticky=E)
predbutton.grid(row=3,column=1,sticky=E)
predAnslabel.grid(row=3,column=2,padx=10,sticky=W)
clfframe.grid(row=2,column=2,padx=5,pady=5)

# 系統回報Widget
sysreport = Text(root,width=25,height=30,wrap='char',borderwidth=1,relief='solid')
sysreport.configure(state='normal')
sysreport.insert('1.0','====系統回報區====\n提示:\n*Enter:連線/中斷連線\n*ESC:離開\n*系統會直接把影像以720x480的尺寸顯示在視窗內，但影像資訊的大小不會改變。\n================\n')
sysreport.configure(state='disabled')
sysreport.grid(row=1,column=0,padx=2,pady=2)

root.mainloop()    


