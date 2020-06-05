# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:44:09 2019

@author: Tim Koorey
"""

from PyQt5 import QtCore, QtGui, QtWidgets

from astropy.io import fits

import numpy as np

import socket

from Either import *
from Parsers import *

#import kicklib as kl

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import socket
import os
import astropy.io.fits as fits
import time
import numpy as np


TCPIP='10.103.154.4'
PORT= 54321
BUFFER_SIZE = 512
TMPFITS= '/home/lab/zdrive/kuroTemp/temp.fit'

class Camera():
    def __init__(self):

        try:
            s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((TCPIP,PORT))
            ml='alive?'
            s.send(ml.encode())
            #a=s.recv(5)
            s.close()

        except:
            print('Could not initialise camera')
        


    def shoot(self,nframes):
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((TCPIP,PORT))
        ml='acquire'+str(nframes)
        s.send(ml.encode())

    def read(self):
        t=0
        while not (os.path.exists(TMPFITS)):
            time.sleep(0.1)
            t=t+1
            if (t>100): break
        hdu=fits.open(TMPFITS)
        imgdata=hdu[0].data
        outdata=np.array(imgdata)
        hdu.close()
        #os.remove(TMPFITS)
        return outdata

    def plot(self,data,npics):
        import matplotlib.pyplot as plt
        realdata=data[0,:,:]
        if (npics==1):
            plt.imshow(realdata)#, cmap=cm.jet, aspect='auto')
            plt.show()

#if __name__=='__main__':
#    import DAQ
#    EDRE=DAQ.EDRE_Interface()
#    cam=camera()
#    cam.shoot(1)
#    time.sleep(0.1)
#    EDRE.writeChannel(0,19,5000000)
#    time.sleep(0.01)
#    EDRE.writeChannel(0,19,0)
#    a=cam.read()
#    cam.plot(a,1)
#    print(a.shape)

camera = Camera()

 
def doCutoff(ratio):       
    if np.isinf(ratio):
        ratio = 1
    elif np.isnan(ratio):
        ratio = 1
    return ratio

def doCutoff_Vec(ratio):
    vec = np.vectorize(doCutoff)
    return vec(ratio)

class MouseGraphicsView(QtWidgets.QGraphicsView):

    pressEvent = QtCore.pyqtSignal(QtCore.QPointF, name = "pressEvent")
    releaseEvent = QtCore.pyqtSignal(QtCore.QPointF, name = "releaseEvent")
    statusEvent = QtCore.pyqtSignal(QtCore.QPointF, name = "statusEvent")
        
    def __init__(self,parent=None):
        super(MouseGraphicsView,self).__init__(parent)
        
    def mousePressEvent(self,event):
        pos = self.mapToScene(event.pos())
        print("Mouse Pressed %d %d\n" % (int(pos.x(),int(pos.y()))))
        self.pressEvent.emit(pos)
        
    def mouseReleaseEvent(self,event):
        pos = self.mapToScene(event.pos())
        print("Mouse Released %d %d\n" % (int(pos.x()),int(pos.y())))
        self.pressEvent.emit(pos)

    def mouseMoveEvent(self,event):
        pos = self.mapToScene(event.pos())
        self.statusEvent.emit(pos)

class ImageSelect(QtWidgets.QWidget):
    def __init__(self, picture, parent = None):
        super(ImageSelect,self).__init__(parent)

        self.picture = picture

        self.parent = parent

        self.gbImageOptions = QtWidgets.QGroupBox(parent)
        self.gbImageOptions.setGeometry(QtCore.QRect(250, 100, 120, 211))
        self.gbImageOptions.setObjectName("gbImageOptions")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.gbImageOptions)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 20, 157, 151))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.rbImage = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.rbImage.setObjectName("rbImage")
        self.verticalLayout.addWidget(self.rbImage)
        self.rbData = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.rbData.setObjectName("rbData")
        self.verticalLayout.addWidget(self.rbData)
        self.rbNoAtoms = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.rbNoAtoms.setObjectName("rbNoAtoms")
        self.verticalLayout.addWidget(self.rbNoAtoms)
        self.rbNoLaser = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.rbNoLaser.setObjectName("rbNoLaser")
        self.verticalLayout.addWidget(self.rbNoLaser)
        self.rbCorrectBackground = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.rbCorrectBackground.setObjectName("rbCorrectBackground")
        self.verticalLayout.addWidget(self.rbCorrectBackground)
        self.cbFilter = QtWidgets.QCheckBox(self.gbImageOptions)
        self.cbFilter.setGeometry(QtCore.QRect(0, 180, 90, 23))
        self.cbFilter.setObjectName("cbFilter")
        self.cbShadow = QtWidgets.QCheckBox(parent)
        self.cbShadow.setGeometry(QtCore.QRect(250, 50, 90, 23))
        self.cbShadow.setObjectName("cbShadow")

        self.rbImage.setChecked(True)
        self.picture.PicShow = 1
        self.cbShadow.setChecked(False)
        self.rbNoLaser.setEnabled(False)
        self.rbCorrectBackground.setEnabled(False)

        self.retranslateUi()
        self.cbShadow.toggled.connect(self.setShadow)
        self.rbImage.toggled.connect(self.setImage)
        self.rbData.toggled.connect(self.setAtoms)
        self.rbNoAtoms.toggled.connect(self.setNoAtoms)
        self.rbNoLaser.toggled.connect(self.setNoLaser)
        self.rbCorrectBackground.toggled.connect(self.setBackground)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "self"))
        self.gbImageOptions.setTitle(_translate("self", "Image Options"))
        self.rbImage.setText(_translate("self", "Image"))
        self.rbData.setText(_translate("self", "Atoms"))
        self.rbNoAtoms.setText(_translate("self", "No Atoms"))
        self.rbNoLaser.setText(_translate("self", "No Laser"))
        self.rbCorrectBackground.setText(_translate("self", "Background"))
        self.cbFilter.setText(_translate("self", "Filter"))
        self.cbShadow.setText(_translate("self", "Shadow"))

    def setShadow(self,state):
        self.picture.IsShadow = state

        if not state:
            self.rbNoLaser.setEnabled(False)
            self.rbCorrectBackground.setEnabled(False)
        else:
            self.rbNoLaser.setEnabled(True)
            self.rbCorrectBackground.setEnabled(True)

    def setImage(self,state):
        if state:
            self.picture.PicShow = 1
            self.picture.plot()

    def setAtoms(self,state):
        if state:
            self.picture.PicShow = 2
            self.picture.plot()

    def setNoAtoms(self,state):
        if state:
            self.picture.PicShow = 3
            self.picture.plot()

    def setNoLaser(self,state):
        if state:
            self.picture.PicShow = 4
            self.picture.plot()

    def setBackground(self,state):
        if state:
            self.picture.PicShow = 5
            self.picture.plot()

class Picture(QtWidgets.QWidget):
    def __init__(self, xs=512, ys=512, dpi=90, parent=None):
        super(Picture,self).__init__(parent)

        self.parent = parent

        self.FirstPlot = True

        self.XSize = xs
        self.YSize = ys
       
        self.data = np.zeros((xs,ys),dtype=np.uint16)
        self.noAtoms = np.zeros((xs,ys),dtype=np.uint16)
        self.noLaser = np.zeros((xs,ys),dtype=np.uint16)

        self.Backgrounds = np.zeros((maximumBackgrounds,xs,ys),dtype=np.uint16)
        self.BackgroundID = 0
        self.numBackgrounds = 0
        self.correctBG = False
        
        self.Display = np.zeros((xs,ys),dtype=float)
        self.CorImage = np.zeros((xs,ys),dtype=float)

        self.ROIx1 = 200
        self.ROIx2 = 350
        self.ROIy1 = 30
        self.ROIy2 = 150
        
        self.numAtoms = 0

        self.isFocussing = False
        self.DisplayLowPass = False

        self.figure =  Figure(figsize=(xs, ys), dpi=dpi) #QtGui.QImage(512,512,QtGui.QImage.Format_RGB32)
        
        self.canvas = FigureCanvas(self.figure)

        self.imWidget = QtWidgets.QWidget(self)
        self.imWidget.setGeometry(0, 0 ,512, 512)
        self.imLayout = QtWidgets.QVBoxLayout(self.imWidget)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.imWidget)
        self.imLayout.addWidget(self.canvas)
        self.imLayout.addWidget(self.mpl_toolbar)
        self.ax = self.figure.add_subplot(111)
#        self.figure.tight_layout()
        self.ax.axis("off")

#        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))
#        self.updateGeometry()
#        self.plot()
        
        
#        self.LUT = readLUT("Fire.lut")
        
        self.PicShow = 2
        self.IsShadow = False

        #self.scene = QtWidgets.QGraphicsScene(self)
        #self.view = MouseGraphicsView(self)
        #self.view.setMouseTracking(True)
        #pixmap = QtGui.QPixmap.fromImage(self.image)
        #self.pixmap = self.scene.addPixmap(pixmap)
        #self.view.setScene(self.scene)
    
    def loadFile(self,fileName):
        hdu_list = fits.open(fileName)
        
        data = hdu_list[0].data
        
        self.data = data[0]
        self.noAtoms = data[1]
        self.noLaser = data[2]
        
        hdu_list.close()

        self.plot()
        
    def loadData(self,data):
        
        self.data = data[0]
        self.noAtoms = data[1]
#        self.noLaser = data[2]

        self.plot()
       
        #Start edit
        
    def GetBG(self):
        pass
        
    def GetImage(self,scale):
        
        MaxVal = -1.0e9
        MinVal = 1.0e9
        MaxXInd = 0
        MaxYInd = 0
        MinXInd = 0
        MinYInd = 0
        intfluo=0.0
        
        if self.PicShow == 5 and self.IsShadow:
            GetBG()
            
        if scale == 0:
        
            if self.IsShadow:
                            
                if self.PicShow == 1:
                    t2 = self.noAtoms - self.noLaser
                    t1 = self.data - self.noLaser
                    if t2 != 0.0:
                        t1 = t1/t2
                    if t1 > 0.0:
                        t1 = -np.log(t1)
                    else:
                        t1 = 0.0

                elif self.PicShow == 2:
                    t1 = self.data
                elif self.PicShow == 3:
                    t1 = self.noAtoms
                elif self.PicShow == 4:
                    t1 = self.noLaser
                elif self.PicShow == 5:
                    t1 = self.noAtoms
                        
            elif self.isFocussing:
                t1 = self.data
                            
            else:
                if self.PicShow == 1:
                    t1 = self.data - self.noAtoms
                elif self.PicShow == 2:
                    t1 = self.data
                elif self.PicShow == 3:
                    t1 = self.noAtoms
                #intfluo = intfluo + t1
                    
            self.Display=t1
            self.Display[self.Display < 0] = 0

        else:
            MaxVal = scale
            MinVal = 2000
            self.Display=self.data
                    
        if self.DisplayLowPass:
            self.Display = kl.filter2(self.Display,self.XSize,self.YSize)
            
        MaxVal = -1e9
        MinVal = 1e9
        
        #for i in range(self.YSize):
            #for j in range(self.XSize):
                #ti=i*self.XSize+j
                
        self.Display[np.isnan(self.Display)] =0.01

        MaxVal = np.amax(self.Display)
        ind = np.where(self.Display==MaxVal)
        MaxXInd = ind[1]
        MaxYInd = ind[0]
            
#        print("Min of %g at %d %d Max of %g at %d %d\n" % (MinVal,MinXInd,MinYInd,MaxVal,MaxXInd,MaxYInd))
        
        contrast = MaxVal-MinVal
    
        if contrast == 0.0:
            contrast = 1.0
            print("Noooo!")
        
        tt1 =self.Display - MinVal
        ratio = tt1/contrast

        cutoff = np.array(list(doCutoff_Vec(ratio)))
        
        tt1=255*ratio
        temp = np.floor(tt1)
                
        
        ROIactive = True #Set to false to disable the following sequence
        
        if ROIactive and self.PicShow == 5:
            
            for i in range(self.ROIx1,self.ROIx2):
                self.data[i,self.ROIy1] = 0x00000000 #self.image.setPixel(i,self.ROIy1,0x00000000)
                self.data[i,self.ROIy2]=0x00000000 #self.image.setPixel(i,self.ROIy2,0x00000000)
            
            for j in range(self.ROIy1,self.ROIy2):
                self.data[j,self.ROIx1] = 0x00000000 #self.image.setPixel(self.ROIx1,j,0x00000000)
                self.data[j,self.ROIx2] = 0x00000000 #self.image.setPixel(self.ROIx2,j,0x00000000)
                
            mysum = 0.0
            for i in range(self.ROIx1,self.ROIx2):
                for j in range(self.ROIy1,self.ROIy2):
                    mysum=mysum+self.Display[j*self.XSize+i]
                
            pixSize = 2.1e-6
            pixArea = pixSize*pixSize
            sigma=1.4e-13
                
            self.numAtoms = np.floor(mysum*pixArea/sigma)
            return MaxVal, self.numAtoms
        
        
    def plot(self):
        self.GetImage(0)
        if self.FirstPlot:
            self.im = self.ax.imshow(self.Display, cmap=cm.jet, aspect='auto')
            self.FirstPlot = False
        else:
            self.im.set_data(self.Display)
            self.figure.canvas.draw()
#        self.canvas.draw()

    
