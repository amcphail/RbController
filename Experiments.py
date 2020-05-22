# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Optimiser.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Optimiser(QtWidgets.QWidget):
    def __init__(self, tabNumber, name, parent=None):
        super(Optimiser,self).__init__(parent)

        self.tabNumber = tabNumber
        self.name = name
        
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(10, 40, 681, 551))
        self.widget.setObjectName("widget")
        self.lbStartTime = QtWidgets.QLabel(self.widget)
        self.lbStartTime.setGeometry(QtCore.QRect(20, 20, 81, 16))
        self.lbStartTime.setObjectName("lbStartTime")
        self.lbStartTime.setText("Start Time:")
        self.sbStartTime = QtWidgets.QSpinBox(self.widget)
        self.sbStartTime.setGeometry(QtCore.QRect(110, 20, 47, 24))
        self.sbStartTime.setMaximum(100000)
        self.sbStartTime.setSingleStep(100)
        self.sbStartTime.setObjectName("sbStartTime")

        self.lbBins = QtWidgets.QLabel(self.widget)
        self.lbBins.setGeometry(QtCore.QRect(280, 30, 51, 16))
        self.lbBins.setObjectName("lbBins")
        self.lbBins.setText("Bins:")
        self.sbBins = QtWidgets.QSpinBox(self.widget)
        self.sbBins.setGeometry(QtCore.QRect(370, 30, 61, 24))
        self.sbBins.setMinimum(1)
        self.sbBins.setProperty("value", 1)
        self.sbBins.setObjectName("sbBins")
        self.lbBinDuration = QtWidgets.QLabel(self.widget)
        self.lbBinDuration.setGeometry(QtCore.QRect(280, 60, 81, 16))
        self.lbBinDuration.setObjectName("lbBinDuration")
        self.lbBinDuration.setText("Bin Duration:")
        self.lbStartTimeUnits = QtWidgets.QLabel(self.widget)
        self.lbStartTimeUnits.setGeometry(QtCore.QRect(170, 30, 21, 16))
        self.lbStartTimeUnits.setObjectName("lbStartTimeUnits")
        self.lbStartTimeUnits.setText("us")
        self.sbBinDuration = QtWidgets.QSpinBox(self.widget)
        self.sbBinDuration.setGeometry(QtCore.QRect(370, 60, 61, 24))
        self.sbBinDuration.setMaximum(1000000)
        self.sbBinDuration.setSingleStep(100)
        self.sbBinDuration.setProperty("value", 1000)
        self.sbBinDuration.setObjectName("sbBinDuration")
        self.lbBinDurationUnits = QtWidgets.QLabel(self.widget)
        self.lbBinDurationUnits.setGeometry(QtCore.QRect(440, 60, 31, 16))
        self.lbBinDurationUnits.setObjectName("lbBinDurationUnits")
        self.lbBinDurationUnits.setText("us")

        self.lbChannels = QtWidgets.QLabel(self.widget)
        self.lbChannels.setGeometry(QtCore.QRect(20, 50, 57, 15))
        self.lbChannels.setObjectName("lbChannels")
        self.lbChannels.setText("Channels:")
        self.sbChannels = QtWidgets.QSpinBox(self.widget)
        self.sbChannels.setGeometry(QtCore.QRect(110, 50, 47, 24))
        self.sbChannels.setMaximum(24)
        self.sbChannels.setProperty("value", 1)
        self.sbChannels.setObjectName("sbChannels")
        self.cbChannels = QtWidgets.QComboBox(self.widget)
        self.cbChannels.setGeometry(QtCore.QRect(20, 90, 131, 23))
        self.cbChannels.setObjectName("cbChannels")

        self.pbAddChannel = QtWidgets.QPushButton(self.widget)
        self.pbAddChannel.setGeometry(QtCore.QRect(170, 90, 91, 23))
        self.pbAddChannel.setObjectName("pbAddChannel")
        self.pbAddChannel.setText("Add Channel")

        self.scrollArea = QtWidgets.QScrollArea(self.widget)
        self.scrollArea.setGeometry(QtCore.QRect(20, 130, 651, 251))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 649, 249))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.tableWidget = QtWidgets.QTableWidget(self.scrollAreaWidgetContents)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 651, 251))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.pbSetInitialGuess = QtWidgets.QPushButton(self.widget)
        self.pbSetInitialGuess.setGeometry(QtCore.QRect(20, 390, 121, 23))
        self.pbSetInitialGuess.setObjectName("pbSetInitialGuess")
        self.pbSetInitialGuess.setText("Set Initial Guess")

        self.pbLoad = QtWidgets.QPushButton(self.widget)
        self.pbLoad.setGeometry(QtCore.QRect(530, 30, 80, 23))
        self.pbLoad.setObjectName("pbLoad")
        self.pbLoad.setText("Load")
        self.pbSave = QtWidgets.QPushButton(self.widget)
        self.pbSave.setGeometry(QtCore.QRect(530, 60, 80, 23))
        self.pbSave.setObjectName("pbSave")
        self.pbSave.setText("Save")

        QtCore.QMetaObject.connectSlotsByName(Form)

    def getName(self):
        return self.name

        
class Flourescence(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Flourescence,self).__init__(parent)
        
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(10, 40, 681, 551))
        self.widget.setObjectName("widget")
        self.lbStartTime = QtWidgets.QLabel(self.widget)
        self.lbStartTime.setGeometry(QtCore.QRect(10, 20, 91, 16))
        self.lbStartTime.setObjectName("lbStartTime")
        self.lbStartTime.setText("Start Time:")
        self.sbStartTime = QtWidgets.QSpinBox(self.widget)
        self.sbStartTime.setGeometry(QtCore.QRect(110, 20, 71, 24))
        self.sbStartTime.setMaximum(1000000)
        self.sbStartTime.setSingleStep(100)
        self.sbStartTime.setObjectName("sbStartTime")
        self.lbImageOneTime = QtWidgets.QLabel(self.widget)
        self.lbImageOneTime.setGeometry(QtCore.QRect(20, 60, 71, 16))
        self.lbImageOneTime.setObjectName("lbImageOneTime")
        self.lbImageOneTime.setText("Image One:")
        self.sbImageOneTime = QtWidgets.QSpinBox(self.widget)
        self.sbImageOneTime.setGeometry(QtCore.QRect(110, 60, 71, 24))
        self.sbImageOneTime.setMaximum(1000000)
        self.sbImageOneTime.setSingleStep(100)
        self.sbImageOneTime.setObjectName("sbImageOneTime")
        self.lbStartTimeUnits = QtWidgets.QLabel(self.widget)
        self.lbStartTimeUnits.setGeometry(QtCore.QRect(190, 20, 21, 16))
        self.lbStartTimeUnits.setObjectName("lbStartTimeUnits")
        self.lbStartTimeUnits.setText("us")
        self.lbImageOneUnits = QtWidgets.QLabel(self.widget)
        self.lbImageOneUnits.setGeometry(QtCore.QRect(190, 60, 21, 16))
        self.lbImageOneUnits.setObjectName("lbImageOneUnits")
        self.lbImageOneUnits.setText("us")
        self.lbImageTwoTime = QtWidgets.QLabel(self.widget)
        self.lbImageTwoTime.setGeometry(QtCore.QRect(20, 100, 71, 16))
        self.lbImageTwoTime.setObjectName("lbImageTwoTime")
        self.lbImageTwoTime.setText("Image Two:")
        self.sbImageTwoTime = QtWidgets.QSpinBox(self.widget)
        self.sbImageTwoTime.setGeometry(QtCore.QRect(110, 100, 71, 24))
        self.sbImageTwoTime.setMaximum(1000000)
        self.sbImageTwoTime.setSingleStep(100)
        self.sbImageTwoTime.setObjectName("sbImageTwoTime")
        self.lbImageTwoTimeUnits = QtWidgets.QLabel(self.widget)
        self.lbImageTwoTimeUnits.setGeometry(QtCore.QRect(190, 100, 21, 16))
        self.lbImageTwoTimeUnits.setObjectName("lbImageTwoTimeUnits")
        self.lbImageTwoTimeUnits.setText("us")

        QtCore.QMetaObject.connectSlotsByName(Form)

    def getName(self):
        return "MOT 2 Flourescence"
        
class Experiments(QtWidgets.QWidget):
    def __init__(self, parent= None):
        super(Experiments,self).__init__(parent)

        self.tbExperimentTab = QtWidgets.QWidget(parent)
        self.tbExperimentTab.setGeometry(QtCore.QRect(10, 10, 700, 600))
        self.tbExperimentTab.setObjectName("tbExperimentTab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(parent)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 671, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pbLoad = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pbLoad.setObjectName("pbLoad")
        self.pbLoad.setText("Load")
        self.horizontalLayout.addWidget(self.pbLoad)
        self.pbSave = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pbSave.setObjectName("pbSave")
        self.pbSave.setText("Save")
        self.horizontalLayout.addWidget(self.pbSave)
        self.horizontalLayout.setStretch(0, 1)
        self.lbExperiment = QtWidgets.QLabel(self.tbExperimentTab)
        self.lbExperiment.setGeometry(QtCore.QRect(20, 90, 151, 16))
        self.lbExperiment.setObjectName("lbExperiment")
        self.lbExperiment.setText("Experiment")
        self.cbModules = QtWidgets.QComboBox(self.tbExperimentTab)
        self.cbModules.setGeometry(QtCore.QRect(470, 90, 181, 23))
        self.cbModules.setObjectName("cbModules")
        self.pbAddModule = QtWidgets.QPushButton(self.tbExperimentTab)
        self.pbAddModule.setGeometry(QtCore.QRect(290, 90, 80, 23))
        self.pbAddModule.setObjectName("pbAddModule")
        self.pbAddModule.setText("Add")
        self.pbRemove = QtWidgets.QPushButton(self.tbExperimentTab)
        self.pbRemove.setGeometry(QtCore.QRect(380, 90, 80, 23))
        self.pbRemove.setObjectName("pbRemove")
        self.pbRemove.setText("Remove")
        self.lwExperiment = QtWidgets.QListWidget(self.tbExperimentTab)
        self.lwExperiment.setGeometry(QtCore.QRect(20, 120, 256, 192))
        self.lwExperiment.setObjectName("lwExperiment")

        QtCore.QMetaObject.connectSlotsByName(Form)
        
class Ui_Form(object):
    def setupUi(self, Form):

        Form.setObjectName("Form")
        Form.resize(728, 636)

        self.num_tabs = 0
        self.tabs = {}
        self.tab_layouts = {}
        self.tab_contents = {}
        
        self.centralwidget = QtWidgets.QWidget(Form)
        self.centralwidget.setObjectName("centralwidget")
        self.tbFancy = QtWidgets.QTabWidget(self.centralwidget)
        self.tbFancy.setGeometry(QtCore.QRect(10, 20, 701, 601))
        self.tbFancy.setObjectName("tbFancy")

        self.addTab(lambda x: Optimiser(x,"Optimiser"))
        self.addTab(Flourescence)
        
#        self.tbExperimentTab = Experiments(self.tbFancy)
#        self.tbExperimentTab.setObjectName("tbExperimentTap")
        
        self.tbFancy.addTab(self.tabs[0], self.tab_contents[0].getName())
        self.tbFancy.addTab(self.tabs[1], self.tab_contents[1].getName())
#        self.tbFancy.addTab(self.tbExperimentTab, "Experiments")

        self.retranslateUi(Form)
        self.tbFancy.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def addTab(self,tab):
        self.tabs[self.num_tabs] = QtWidgets.QWidget(self.tbFancy)
        self.tab_layouts[self.num_tabs] = QtWidgets.QVBoxLayout(self.tabs[self.num_tabs])
        self.tabs[self.num_tabs].setObjectName("tbTab"+str(self.num_tabs))
        self.tabs[self.num_tabs].setLayout(self.tab_layouts[self.num_tabs])
        
        self.tab_contents[self.num_tabs] = tab(self.tabs[self.num_tabs])
        self.tab_contents[self.num_tabs].setObjectName("Payload"+str(self.num_tabs))

        self.tab_layouts[self.num_tabs].addWidget(self.tab_contents[self.num_tabs])

        self.num_tabs += 1
        
        
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

