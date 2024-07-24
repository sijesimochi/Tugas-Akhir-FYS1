# ----- Imports -------------------------------------------------------

# Standard Imports
import sys
import numpy as np
import time
import math
import struct
import os
import string
import serial
import serial.tools.list_ports
import statistics
import warnings
import random
import copy
import collections
import pandas as pd
import datetime
import threading
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

# import firebase modules
import firebase_admin
from firebase_admin import db, credentials, messaging

base_dir = "ABSOLUTE/PATH/TO/THIS/DIRECTORY"
cred = credentials.Certificate(os.path.join(base_dir, "YOURAPIKEYHERE"))
firebase_admin.initialize_app(
    cred, {"databaseURL": "YOURDATABASEURLHERE"}
)
ref = db.reference("/")

# PyQt5 Imports
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDial,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QButtonGroup,
    QFormLayout,
    QFrame,
    QSpacerItem,
)
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.pgcollections import OrderedDict
from collections import deque
from gl_classes import GLTextItem

# Local File Imports
from gui_threads import *
from gui_parser import uartParser
from graphUtilities import *
from gui_common import *
from cachedData import *
# from fall_detection import FallDetection, fallDetectionSliderClass
from buzzer import *


# ----- Defines -------------------------------------------------------
compileGui = 0
# Define column names
firstColumns = [
    (
        [
            "TS",
            "X-Pos",
            "Y-Pos",
            "Z-Pos",
            "X-Vel",
            "Y-Vel",
            "Z-Vel",
            "X-Acc",
            "Y-Acc",
            "Z-Acc",
        ]
    )
]
columns = [
    "TS",
    "X-Pos",
    "Y-Pos",
    "Z-Pos",
    "X-Vel",
    "Y-Vel",
    "Z-Vel",
    "X-Acc",
    "Y-Acc",
    "Z-Acc",
]
fileName = "rawData10.csv"

# df_new = pd.DataFrame(firstColumns, columns=columns, copy=False)
# df_new.to_csv(fileName, mode='a', index=False, header=False)

# Create an empty DataFrame
# df_new = pd.DataFrame(columns=columns)

# Save DataFrame to Excel initially
# df.to_csv('rawData.csv', index=False)

# global fallCon, xPos, yPos, zPos
fallCon = False
fatalCon = False
xPos = 0
yPos = 0
zPos = 0
detectObject = False
fallConDisplay = 0
subjectStatus = "0"
prediction = 0
rawDataToModel = []
oneBatch = []
start_time = time.time()  # Current time in seconds since the epoch
df = [
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
]
sendJatuh = True
sendFatal = True

# CachedData holds the data from the last configuration run for faster prototyping and testing
cachedData = cachedDataType()
# Only when compiling
if compileGui:
    from fbs_runtime.application_context.PyQt5 import ApplicationContext


# Preprocess Data
def fall_det(df):
    ## Transform Mean
    mean = np.mean(df, axis=0)

    ## Transform Variance
    var = np.var(df, axis=0)

    ## Gabungin jadi satu data frame
    merged = np.concatenate((mean, var)).reshape(1, -1)
    merged = pd.DataFrame(merged)

    ## Load model
    fall_det = joblib.load(os.path.join(base_dir, "Model/All 9 Features/svm.pkl"))
    prediction = fall_det.predict(merged)

    return prediction


# Create a list of N distict colors, visible on the black GUI background, for our tracks
# The format for a single color is (r,g,b,a) -> normalized from 0-255 to 0-1
# LUT based on Kelly's 22 Colors of Max Contrast, slightly adjusted for better visibility on black background (https://sashamaps.net/docs/resources/20-colors/)
# Only the first 21 colors are guaranteed to be highly distinct. After that colors are generated, but not promised to be visually distinct.
def get_trackColors(n):
    # Modified LUT of Kelly's 22 Colors of Max Contrast
    modKellyColors = [
        # (255, 255, 255, 255),   # White
        # (  0,   0,   0, 255),   # Black
        # (169, 169, 169, 255),   # Gray
        (230, 25, 75, 255),  # Red
        (60, 180, 75, 255),  # Green
        (255, 225, 25, 255),  # Yellow
        (67, 99, 216, 255),  # Blue
        (245, 130, 49, 255),  # Orange
        (145, 30, 180, 255),  # Purple
        (66, 212, 244, 255),  # Cyan
        (240, 50, 230, 255),  # Magenta
        (191, 239, 69, 255),  # Lime
        (250, 190, 212, 255),  # Pink
        (70, 153, 144, 255),  # Teal
        (220, 190, 255, 255),  # Lavender
        (154, 99, 36, 255),  # Brown
        (255, 250, 200, 255),  # Beige
        (128, 0, 0, 255),  # Maroon
        (170, 255, 195, 255),  # Mint
        (128, 128, 0, 255),  # Olive
        (255, 216, 177, 255),  # Apricot
        (0, 0, 117, 255),  # Navy
    ]

    # Generate normalized version of Kelly colors
    modKellyColorsNorm = []
    for tup in modKellyColors:
        modKellyColorsNorm.append(tuple(ti / 255 for ti in tup))

    # Create the output color list
    trackColorList = []
    for i in range(n):
        # If within the length of the LUT, just grab values
        if i < len(modKellyColorsNorm):
            trackColorList.append(modKellyColorsNorm[0])
        # Otherwise, generate a color from the average of two randomly selected colors, and add the new color to the list
        else:
            (r_2, g_2, b_2, _) = modKellyColorsNorm[
                random.randint(0, len(modKellyColorsNorm) - 1)
            ]
            (r_1, g_1, b_1, _) = modKellyColorsNorm[
                random.randint(0, len(modKellyColorsNorm) - 1)
            ]
            r_gen = (r_2 + r_1) / 2
            g_gen = (g_2 + g_1) / 2
            b_gen = (b_2 + b_1) / 2
            modKellyColorsNorm.append((r_gen, g_gen, b_gen, 1.0))
            trackColorList.append((r_gen, g_gen, b_gen, 1.0))

    return trackColorList


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def visualizePointCloud(heights, tracks, self):
    global fallCon, xPos, yPos, zPos, detectObject, subjectStatus, df
    if heights is not None:
        detectObject = True
        # subjectStatus = "1"
        # print(heights)
        if len(heights) != len(tracks):
            print("WARNING: number of heights does not match number of tracks")
        # Compute the fall detection results for each object
        # fallDetectionDisplayResults = self.fallDetection.step(heights, tracks)
        ## Display fall detection results

        # For each height heights for current tracks
        sent = []
        for height in heights:
            # Find track with correct TID
            for track in tracks:
                # Found correct track
                if int(track[0]) == int(height[0]):
                    tid = int(height[0])
                    height_str = "Height : " + str(round(height[1], 2)) + " m"
                    # ts store timestamp of current time
                    ct = datetime.datetime.now()
                    ts = ct.timestamp()

                    rawData = track[1:10]
                    rawDataToModel = track[1:10]
                    if len(oneBatch) >= 18:
                        oneBatch.pop(0)
                    oneBatch.append(rawDataToModel)
                    df = pd.DataFrame(oneBatch)
                    print(df)

                    rawData = np.insert(rawData, 0, ts)
                    xPos = track[1]
                    yPos = track[2]
                    zPos = track[3]
                    # print(rawData)
                    # sent.append(ts)
                    sent.append(rawData)
                    # print(sent)
                    # print("Xpos:" + str(xPos) + " Ypos:" + str(yPos) + " Zpos:" + str(zPos))
                    # print(trackIndexs)
                    # print(numPoints)
                    # for i in range(numPoints):
                    #     print(pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], pointCloud[i,3])
                    # If this track was computed to have fallen, display it on the screen

                    # Append the new data to the DataFrame
                    # df_new = pd.DataFrame(sent, columns=columns, copy=False)

                    # Save the updated DataFrame to Excel
                    # df_new.to_csv(fileName, mode='a', index=False, header=False)

                    if prediction == 1:
                        on_all()
                        fallCon = True
                        height_str = height_str + " FALL DETECTED"
                        # print("jatuh")
                        if fatalCon == True:
                            subjectStatus = "3"
                            self.subjectSetupImg = QPixmap(
                                os.path.join(base_dir, "images/4Small.png")
                            )
                        else:
                            subjectStatus = "2"
                            self.subjectSetupImg = QPixmap(
                                os.path.join(base_dir, "images/3Small.png")
                            )
                    else:
                        off_all()
                        fallCon = False
                        # print("tidak jatuh")
                        subjectStatus = "1"
                        self.subjectSetupImg = QPixmap(
                            os.path.join(base_dir, "images/2Small.png")
                        )

                    self.subjectImgLabel.setPixmap(self.subjectSetupImg)
                    self.coordStr[tid].setText(height_str)
                    self.coordStr[tid].setX(track[1])
                    self.coordStr[tid].setY(track[2])
                    self.coordStr[tid].setZ(track[3])
                    self.coordStr[tid].setVisible(True)
                    self.plotXPos.setText("X-Pos: " + str(xPos))
                    self.plotYPos.setText("Y-Pos: " + str(yPos))
                    self.plotZPos.setText("Z-Pos: " + str(zPos))
                    break
    else:
        if fallCon == True:
            on_all()
            subjectStatus = "2"
            self.subjectSetupImg = QPixmap(os.path.join(base_dir, "images/3Small.png"))
        else:
            off_all()
            xPos = 0
            yPos = 0
            zPos = 0
            subjectStatus = "0"
            self.subjectSetupImg = QPixmap(os.path.join(base_dir, "images/1Small.png"))

        self.plotXPos.setText("X-Pos: " + str(xPos))
        self.plotYPos.setText("Y-Pos: " + str(yPos))
        self.plotZPos.setText("Z-Pos: " + str(zPos))
        self.subjectImgLabel.setPixmap(self.subjectSetupImg)


def sentToFirebase():
    global xPos, yPos, zPos, detectObject, subjectStatus, sendJatuh, sendFatal, start_time

    current_time = time.time()
    timestamp = (
        time.strftime("%H:%M:%S", time.localtime(current_time))
        .lstrip("0")
        .replace(" 0", " ")
    )
    print(f"Timestamp: {timestamp}")
    db.reference("/timestamp").set(timestamp)

    db.reference("/subjectStatus").set(subjectStatus)
    db.reference("/xPos").set(xPos)
    db.reference("/yPos").set(yPos)
    db.reference("/zPos").set(zPos)

    ref = db.reference("token")
    tokens = ref.get()
    # print(tokens)

    # Define the message
    messageJatuh = messaging.Message(
        notification=messaging.Notification(
            title="FALL Detected!!", body="Check your family now"
        ),
        token=tokens,
        android=messaging.AndroidConfig(priority="high"),
    )
    messageJatuhFatal = messaging.Message(
        notification=messaging.Notification(
            title="FATAL FALL!!", body="Check your family immediatly"
        ),
        token=tokens,
        android=messaging.AndroidConfig(priority="high"),
    )

    # Send the jatuh message
    if fallCon == True:
        if sendJatuh == False:
            sendJatuh = True
            response = messaging.send(messageJatuh)
            print("Successfully sent message:", response)
    else:
        sendJatuh = False

    # Send the jatuh fatal message
    if fatalCon == True:
        if sendFatal == False:
            sendFatal = True
            response = messaging.send(messageJatuhFatal)
            print("Successfully sent message:", response)
    else:
        sendFatal = False


def predictModel():
    global prediction, df
    prediction = fall_det(df)


def predictFatalFall():
    global fatalCon
    if fallCon == True:
        time.sleep(20)
        if fallCon == True:
            fatalCon = True
        else:
            fatalCon = False
    else:
        fatalCon = False
    # print(fatalCon)


class Window(QDialog):
    def __init__(self, parent=None, size=[]):
        super(Window, self).__init__(parent)
        # set window toolbar options, and title
        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("bathMateGUI")

        if (
            0
        ):  # set to 1 to save terminal output to logFile, set 0 to show terminal output
            ts = time.localtime()
            terminalFileName = str(
                "logData/logfile_"
                + str(ts[2])
                + str(ts[1])
                + str(ts[0])
                + "_"
                + str(ts[3])
                + str(ts[4])
                + ".txt"
            )
            sys.stdout = open(terminalFileName, "w")

        print("Python is ", struct.calcsize("P") * 8, " bit")
        print("Python version: ", sys.version_info)

        # TODO bypass serial read function to also log to a file

        self.frameTime = 50
        self.graphFin = 1
        self.hGraphFin = 1
        self.threeD = 1
        self.lastFramePoints = np.zeros((5, 1))
        self.plotTargets = 1
        self.frameNum = 0
        self.profile = {
            "startFreq": 60.25,
            "numLoops": 64,
            "numTx": 3,
            "sensorHeight": 3,
            "maxRange": 10,
            "az_tilt": 0,
            "elev_tilt": 0,
            "enabled": 0,
        }
        self.chirpComnCfg = {
            "DigOutputSampRate": 23,
            "DigOutputBitsSel": 0,
            "DfeFirSel": 0,
            "NumOfAdcSamples": 128,
            "ChirpTxMimoPatSel": 4,
            "ChirpRampEndTime": 36.1,
            "ChirpRxHpfSel": 1,
        }
        self.chirpTimingCfg = {
            "ChirpIdleTime": 8,
            "ChirpAdcSkipSamples": 24,
            "ChirpTxStartTime": 0,
            "ChirpRfFreqSlope": 47.95,
            "ChirpRfFreqStart": 60,
        }
        self.guiMonitor = {
            "pointCloud": 1,
            "rangeProfile": 0,
            "NoiseProfile": 0,
            "rangeAzimuthHeatMap": 0,
            "rangeDopplerHeatMap": 0,
            "statsInfo": 0,
        }
        self.rangeRes = 0
        self.rangeAxisVals = np.zeros(int(self.chirpComnCfg["NumOfAdcSamples"] / 2))
        self.sensorHeight = 1.5
        self.numFrameAvg = 10
        self.configSent = 0
        self.previousFirstZ = -1
        self.yzFlip = 0

        self.trackColorMap = None
        self.prevConfig = DEMO_NAME_3DPC
        self.vitalsPatientData = []
        self.pointBounds = {
            "enabled": False,
            "minX": 0,
            "maxX": 0,
            "minY": 0,
            "maxY": 0,
            "minZ": 0,
            "maxZ": 0,
        }

        # Flag to indicate if the last frame was parsed incorrectly to ignore the error on the subsequent frame when
        # the number of points won't be consistent with the tracker data
        self.lastFrameErrorFlag = False
        # color gradients
        # TODO Simplify color gradients
        self.Gradients = OrderedDict(
            [
                (
                    "bw",
                    {
                        "ticks": [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))],
                        "mode": "rgb",
                    },
                ),
                (
                    "hot",
                    {
                        "ticks": [
                            (0.3333, (185, 0, 0, 255)),
                            (0.6666, (255, 220, 0, 255)),
                            (1, (255, 255, 255, 255)),
                            (0, (0, 0, 0, 255)),
                        ],
                        "mode": "rgb",
                    },
                ),
                (
                    "jet",
                    {
                        "ticks": [
                            (1, (166, 0, 0, 255)),
                            (0.32247191011235954, (0, 255, 255, 255)),
                            (0.11348314606741573, (0, 68, 255, 255)),
                            (0.6797752808988764, (255, 255, 0, 255)),
                            (0.902247191011236, (255, 0, 0, 255)),
                            (0.0, (0, 0, 166, 255)),
                            (0.5022471910112359, (0, 255, 0, 255)),
                        ],
                        "mode": "rgb",
                    },
                ),
                (
                    "summer",
                    {
                        "ticks": [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))],
                        "mode": "rgb",
                    },
                ),
                (
                    "space",
                    {
                        "ticks": [
                            (0.562, (75, 215, 227, 255)),
                            (0.087, (255, 170, 0, 254)),
                            (0.332, (0, 255, 0, 255)),
                            (0.77, (85, 0, 255, 255)),
                            (0.0, (255, 0, 0, 255)),
                            (1.0, (255, 0, 127, 255)),
                        ],
                        "mode": "rgb",
                    },
                ),
                (
                    "winter",
                    {
                        "ticks": [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))],
                        "mode": "rgb",
                    },
                ),
                (
                    "spectrum2",
                    {
                        "ticks": [(1.0, (255, 0, 0, 255)), (0.0, (255, 0, 255, 255))],
                        "mode": "hsv",
                    },
                ),
                (
                    "heatmap",
                    {
                        "ticks": [(1, (255, 0, 0, 255)), (0, (131, 238, 255, 255))],
                        "mode": "hsv",
                    },
                ),
            ]
        )
        cmap = "heatmap"
        if cmap in self.Gradients:
            self.gradientMode = self.Gradients[cmap]
        self.zRange = [-3, 3]
        self.plotHeights = 1
        # Gui size
        if size:
            left = 50
            top = 50
            width = math.ceil(size.width() * 0.8)
            height = math.ceil(size.height() * 0.7)
            self.setGeometry(left, top, width, height)
        # Persistent point cloud
        self.previousClouds = []

        self.hearPlotData = []
        self.breathPlotData = []

        # Set up graph pyqtgraph
        self.init3dGraph()
        self.initColorGradient()
        self.init1dGraph()

        # Add connect options
        self.initConnectionPane()
        self.initStatsPane()
        self.initPlotControlPane()
        self.fallCondition()
        self.initConfigPane()
        self.initSensorPositionPane()
        self.initBoundaryBoxPane()
        # self.initVitalsPlots()

        # Set the layout
        # Create tab for different graphing options
        self.graphTabs = QTabWidget()
        self.graphTabs.addTab(self.pcplot, "Subject Point")
        self.graphTabs.currentChanged.connect(self.whoVisible)

        self.gridlay = QGridLayout()
        self.gridlay.addWidget(self.comBox, 0, 0, 1, 1)
        self.gridlay.addWidget(self.statBox, 1, 0, 1, 1)
        self.gridlay.addWidget(self.configBox, 2, 0, 1, 1)
        self.gridlay.addWidget(self.subjectStatusBox, 3, 0, 1, 1)
        self.gridlay.setRowStretch(7, 1)  # Added to preserve spacing
        self.gridlay.addWidget(self.graphTabs, 0, 1, 8, 1)
        self.gridlay.addWidget(self.colorGradient, 0, 2, 8, 1)
        # self.gridlay.addWidget(self.vitalsPane, 0, 3, 8, 1)

        # self.vitalsPane.setVisible(False)
        self.gridlay.setColumnStretch(0, 1)
        self.gridlay.setColumnStretch(1, 3)
        self.setLayout(self.gridlay)

        # Set up parser
        self.parser = uartParser(type=self.configType.currentText())

        # Check cached data for previously used demo and device to set as default options
        deviceName = cachedData.getCachedDeviceName()
        if deviceName != "":
            try:
                self.deviceType.setCurrentIndex(DEVICE_LIST.index(deviceName))
                if deviceName == "IWR6843":
                    self.parserType = "DoubleCOMPort"
            except:
                print("Device not found. Using default option")
                self.deviceType.setCurrentIndex(0)
        demoName = cachedData.getCachedDemoName()
        if demoName != "":
            try:
                if self.deviceType.currentText() in DEVICE_LIST:
                    self.configType.setCurrentIndex(x843_DEMO_TYPES.index(demoName))

            except:
                print("Demo not found. Using default option")
                self.configType.setCurrentIndex(0)

    def initConnectionPane(self):
        self.comBox = QGroupBox("Connect to Com Ports")
        self.cliCom = QLineEdit("")
        self.dataCom = QLineEdit("")
        self.connectStatus = QLabel("Not Connected")
        self.connectButton = QPushButton("Connect")
        self.saveBinaryBox = QCheckBox("Save UART")
        self.connectButton.clicked.connect(self.connectCom)
        self.configType = QComboBox()
        self.deviceType = QComboBox()

        # TODO Add fall detection support
        # TODO Add replay support
        self.configType.addItems(x843_DEMO_TYPES)
        self.configType.currentIndexChanged.connect(self.onChangeConfigType)
        self.deviceType.currentIndexChanged.connect(self.onChangeDeviceType)
        self.comLayout = QGridLayout()
        self.comLayout.addWidget(QLabel("CLI COM:"), 1, 0)
        self.comLayout.addWidget(self.cliCom, 1, 1)
        self.comLayout.addWidget(QLabel("DATA COM:"), 2, 0)
        self.comLayout.addWidget(self.dataCom, 2, 1)
        self.comLayout.addWidget(self.connectButton, 4, 0)
        self.comLayout.addWidget(self.connectStatus, 4, 1)
        # self.comLayout.addWidget(self.saveBinaryBox, 5, 0)
        self.saveBinaryBox.stateChanged.connect(self.saveBinaryBoxChanged)

        self.comBox.setLayout(self.comLayout)
        self.configType.setCurrentIndex(0)  # initialize this to a stable value

        # Find all Com Ports
        serialPorts = list(serial.tools.list_ports.comports())

        # Find default CLI Port and Data Port
        for port in serialPorts:
            if (
                CLI_XDS_SERIAL_PORT_NAME in port.description
                or CLI_SIL_SERIAL_PORT_NAME in port.description
            ):
                print(f"/tCLI COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.cliCom.setText(comText)

            elif (
                DATA_XDS_SERIAL_PORT_NAME in port.description
                or DATA_SIL_SERIAL_PORT_NAME in port.description
            ):
                print(f"/tData COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.dataCom.setText(comText)

    def initStatsPane(self):
        self.statBox = QGroupBox("Statistics")
        self.frameNumDisplay = QLabel("Frame: 0")
        self.numPointsDisplay = QLabel("Points: 0")
        self.plotXPos = QLabel("X-Pos: 0")
        self.plotYPos = QLabel("Y-Pos: 0")
        self.plotZPos = QLabel("Z-Pos: 0")
        self.statsLayout = QVBoxLayout()
        self.statsLayout.addWidget(self.frameNumDisplay)
        self.statsLayout.addWidget(self.numPointsDisplay)
        self.statsLayout.addWidget(self.plotXPos)
        self.statsLayout.addWidget(self.plotYPos)
        self.statsLayout.addWidget(self.plotZPos)
        self.statBox.setLayout(self.statsLayout)

    def fallDetDisplayChanged(self, newState):
        if newState == 2:
            self.fallDetectionOptionsBox.setVisible(True)
        else:
            self.fallDetectionOptionsBox.setVisible(False)

    def saveBinaryBoxChanged(self, newState):
        if newState == 2:
            self.parser.setSaveBinary(True)
        else:
            self.parser.setSaveBinary(False)

    def fallCondition(self):
        self.subjectStatusBox = QGroupBox("Subject Status")
        self.subjectSetupGrid = QGridLayout()
        self.subjectImgLabel = QLabel()
        self.subjectSetupImg = QPixmap(os.path.join(base_dir, "images/1Small.png"))
        self.subjectSetupGrid.addWidget(self.subjectImgLabel, 1, 1)
        self.subjectImgLabel.setPixmap(self.subjectSetupImg)
        self.subjectStatusBox.setLayout(self.subjectSetupGrid)

    def initPlotControlPane(self):
        self.plotControlBox = QGroupBox("Plot Controls")
        self.pointColorMode = QComboBox()
        self.pointColorMode.addItems(
            [COLOR_MODE_SNR, COLOR_MODE_HEIGHT, COLOR_MODE_DOPPLER, COLOR_MODE_TRACK]
        )
        self.plotTracks = QCheckBox("Plot Tracks")
        self.displayFallDet = QCheckBox("Detect Falls")
        self.persistentFramesInput = QComboBox()
        self.persistentFramesInput.addItems(
            [str(i) for i in range(1, MAX_PERSISTENT_FRAMES + 1)]
        )
        self.persistentFramesInput.setCurrentIndex(2)
        self.plotControlLayout = QFormLayout()
        self.plotControlLayout.addRow("Color Points By:", self.pointColorMode)
        self.plotControlLayout.addRow(self.plotTracks, self.displayFallDet)
        self.plotControlLayout.addRow(
            "# of Persistent Frames", self.persistentFramesInput
        )
        self.plotControlBox.setLayout(self.plotControlLayout)
        # Initialize button values
        self.plotTracks.setChecked(True)

    def updateFallDetectionSensitivity(self):
        self.fallDetection.setFallSensitivity(
            ((self.fallDetSlider.value() / self.fallDetSlider.maximum()) * 0.4) + 0.4
        )  # Range from 0.4 to 0.8

    def initConfigPane(self):
        self.configBox = QGroupBox("Configuration")
        self.selectConfig = QPushButton("Select Configuration")
        self.sendConfig = QPushButton("Start Radar!")
        # self.start = QPushButton("Start without Send Configuration ")
        self.selectConfig.clicked.connect(self.selectCfg)
        self.sendConfig.clicked.connect(self.sendCfg)
        # self.start.clicked.connect(self.startApp)
        self.configLayout = QGridLayout()
        self.configLayout.addWidget(self.selectConfig, 0, 0)
        self.configLayout.addWidget(self.sendConfig, 0, 1)
        # self.configLayout.addWidget(self.start)
        # self.configLayout.addStretch(1)
        self.configBox.setLayout(self.configLayout)

    def setControlLayout(self):
        self.controlBox = QGroupBox("Control")
        self.rangecfar = QSlider(Qt.Horizontal)
        self.azcfar = QSlider(Qt.Horizontal)
        self.snrthresh = QSlider(Qt.Horizontal)
        self.pointsthresh = QSlider(Qt.Horizontal)
        self.gatinggain = QSlider(Qt.Horizontal)
        self.controlLayout = QVBoxLayout()
        self.rangelabel = QLabel("Range CFAR Threshold: ")
        self.azlabel = QLabel("Azimuth CFAR Threshold: ")
        self.snrlabel = QLabel("SNR Threshold: ")
        self.pointslabel = QLabel("Points Threshold: ")
        self.gatinglabel = QLabel("Gating Gain: ")
        self.controlLayout.addWidget(self.rangelabel)
        self.controlLayout.addWidget(self.rangecfar)
        self.controlLayout.addWidget(self.azlabel)
        self.controlLayout.addWidget(self.azcfar)
        self.controlLayout.addWidget(self.snrlabel)
        self.controlLayout.addWidget(self.snrthresh)
        self.controlLayout.addWidget(self.pointslabel)
        self.controlLayout.addWidget(self.pointsthresh)
        self.controlLayout.addWidget(self.gatinglabel)
        self.controlLayout.addWidget(self.gatinggain)
        self.controlBox.setLayout(self.controlLayout)

    # Boundary box control section
    def setBoxControlLayout(self, name):
        # Set up one boundary box control
        boxControl = QGroupBox(name)

        description = QLabel("")
        # Input boxes
        lx = QLineEdit("-6")
        rx = QLineEdit("6")
        ny = QLineEdit("0")
        fy = QLineEdit("6")
        bz = QLineEdit("-6")
        tz = QLineEdit("6")
        enable = QCheckBox()

        # Set up color options
        color = QComboBox()
        color.addItem("Blue", "b")
        color.addItem("Red", "r")
        color.addItem("Green", "g")
        color.addItem("Yellow", "y")
        color.addItem("Cyan", "c")
        color.addItem("Magenta", "m")
        # color.addItem('Black', 'k')
        color.addItem("White", "w")

        boxConLayout = QGridLayout()

        boxConLayout.addWidget(QLabel("Description:"), 0, 0, 1, 1)
        boxConLayout.addWidget(description, 0, 1, 1, 2)
        boxConLayout.addWidget(QLabel("Left X"), 1, 0, 1, 1)
        boxConLayout.addWidget(lx, 1, 1, 1, 1)
        boxConLayout.addWidget(QLabel("Right X"), 1, 2, 1, 1)
        boxConLayout.addWidget(rx, 1, 3, 1, 1)
        boxConLayout.addWidget(QLabel("Near Y"), 2, 0, 1, 1)
        boxConLayout.addWidget(ny, 2, 1, 1, 1)
        boxConLayout.addWidget(QLabel("Far Y"), 2, 2, 1, 1)
        boxConLayout.addWidget(fy, 2, 3, 1, 1)
        boxConLayout.addWidget(QLabel("Bottom Z"), 3, 0, 1, 1)
        boxConLayout.addWidget(bz, 3, 1, 1, 1)
        boxConLayout.addWidget(QLabel("Top Z"), 3, 2, 1, 1)
        boxConLayout.addWidget(tz, 3, 3, 1, 1)
        boxConLayout.addWidget(QLabel("Color"), 4, 0, 1, 1)
        boxConLayout.addWidget(color, 4, 1, 1, 1)
        boxConLayout.addWidget(QLabel("Enable Box"), 4, 2, 1, 1)
        boxConLayout.addWidget(enable, 4, 3, 1, 1)
        boxControl.setLayout(boxConLayout)
        boundList = [lx, rx, ny, fy, bz, tz]

        # Connect onchange listeners
        for text in boundList:
            text.textEdited.connect(self.onChangeBoundaryBox)
        enable.stateChanged.connect(self.onChangeBoundaryBox)
        color.currentIndexChanged.connect(self.onChangeBoundaryBox)
        # Return dictionary of all related controls for this box
        return {
            "name": name,
            "boxCon": boxControl,
            "boundList": boundList,
            "checkEnable": enable,
            "description": description,
            "color": color,
        }

    def initSensorPositionPane(self):
        self.az_tilt = QLineEdit("0")
        self.elev_tilt = QLineEdit("0")
        self.s_height = QLineEdit(str(self.profile["sensorHeight"]))
        self.spLayout = QGridLayout()

        self.spLayout.addWidget(QLabel("Azimuth Tilt"), 0, 0, 1, 1)
        self.spLayout.addWidget(self.az_tilt, 0, 1, 1, 1)
        self.spLayout.addWidget(QLabel("Elevation Tilt"), 1, 0, 1, 1)
        self.spLayout.addWidget(self.elev_tilt, 1, 1, 1, 1)
        self.spLayout.addWidget(QLabel("Sensor Height"), 2, 0, 1, 1)
        self.spLayout.addWidget(self.s_height, 2, 1, 1, 1)

        self.spBox = QGroupBox("Sensor Position")
        self.spBox.setLayout(self.spLayout)
        self.s_height.textEdited.connect(self.onChangeSensorPosition)
        self.az_tilt.textEdited.connect(self.onChangeSensorPosition)
        self.elev_tilt.textEdited.connect(self.onChangeSensorPosition)
        # Force an update so that sensor is at default postion
        self.onChangeSensorPosition()

    def onChangeConfigType(self):
        newConfig = self.configType.currentText()
        cachedData.setCachedDemoName(newConfig)
        print("Demo Changed to: " + newConfig)

        # First, undo any changes that the last demo made
        # These should be the inverse of the changes made in 2nd part of this function

        if self.prevConfig == DEMO_NAME_3DPC:
            self.pointColorMode.setCurrentText(COLOR_MODE_SNR)
            self.displayFallDet.setChecked(True)
            self.displayFallDet.setDisabled(False)
        if newConfig == DEMO_NAME_3DPC:
            self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
            self.displayFallDet.setDisabled(False)
        self.prevConfig = newConfig

    # Callback function to reset settings when device is changed
    def onChangeDeviceType(self):
        newDevice = self.deviceType.currentText()
        cachedData.setCachedDeviceName(newDevice)
        print("Device Changed to: " + newDevice)
        # if newDevice in DEVICE_LIST[0:2]:
        if newDevice in DEVICE_LIST:
            self.configType.currentIndexChanged.disconnect()
            self.dataCom.setEnabled(True)
            self.configType.clear()
            self.configType.addItems(x843_DEMO_TYPES)
            self.parser.parserType = (
                "DoubleCOMPort"  # DoubleCOMPort refers to xWRx843 parts
            )
            self.configType.setCurrentIndex(-1)
            self.configType.currentIndexChanged.connect(self.onChangeConfigType)
            self.configType.setCurrentIndex(0)

    # Gets called whenever the sensor position box is modified
    def onChangeSensorPosition(self):
        try:
            newHeight = float(self.s_height.text())
            newAzTilt = float(self.az_tilt.text())
            newElevTilt = float(self.elev_tilt.text())
        except:
            print("Error in gui_main.py: Failed to update sensor position")
            return
        command = (
            "sensorPosition "
            + self.s_height.text()
            + " "
            + self.az_tilt.text()
            + " "
            + self.elev_tilt.text()
            + " /n"
        )
        # self.cThread = sendCommandThread(self.parser,command)
        # self.cThread.start(priority=QThread.HighestPriority-2)

        # Update Profile info
        self.profile["sensorHeight"] = newHeight

        # Move evmBox to new position
        self.evmBox.resetTransform()
        self.evmBox.rotate(-1 * newElevTilt, 1, 0, 0)
        self.evmBox.rotate(-1 * newAzTilt, 0, 0, 1)
        self.evmBox.translate(0, 0, newHeight)

    def initBoundaryBoxPane(self):
        # Set up all boundary box controls
        self.boundaryBoxes = []
        self.boxTab = QTabWidget()
        self.addBoundBox("pointBounds")

    # For live tuning when available
    def onChangeBoundaryBox(self):
        index = 0
        for box in self.boundaryBoxes:
            # Update dimensions of box
            try:
                xl = float(box["boundList"][0].text())
                xr = float(box["boundList"][1].text())
                yl = float(box["boundList"][2].text())
                yr = float(box["boundList"][3].text())
                zl = float(box["boundList"][4].text())
                zr = float(box["boundList"][5].text())

                boxLines = getBoxLines(xl, yl, zl, xr, yr, zr)
                # boxColor = pg.glColor(
                #     box["color"].itemData(box["color"].currentIndex())
                # )
                boxColor = pg.glColor("c")
                self.boundaryBoxViz[index].setData(
                    pos=boxLines, color=boxColor, width=2, antialias=True, mode="lines"
                )
                # Update visibility
                if box["checkEnable"].isChecked():
                    self.boundaryBoxViz[index].setVisible(True)
                    if "pointBounds" in box["name"]:
                        self.pointBounds["enabled"] = True
                        self.pointBounds["minX"] = xl
                        self.pointBounds["maxX"] = xr
                        self.pointBounds["minY"] = yl
                        self.pointBounds["maxY"] = yr
                        self.pointBounds["minZ"] = zl
                        self.pointBounds["maxZ"] = zr

                else:
                    self.boundaryBoxViz[index].setVisible(False)
                    if "pointBounds" in box["name"]:
                        self.pointBounds["enabled"] = False

                index = index + 1
            except:
                # You get here if you enter an invalid number
                # When you enter a minus sign for a negative value, you will end up here before you type the full number
                pass

    def initColorGradient(self):
        self.colorGradient = pg.GradientWidget(orientation="right")
        self.colorGradient.restoreState(self.gradientMode)
        self.colorGradient.setVisible(False)

    def init3dGraph(self):
        # Create plot
        self.pcplot = gl.GLViewWidget()
        # Sets background to a pastel grey
        self.pcplot.setBackgroundColor(70, 72, 79)
        # Create the background grid
        self.gz = gl.GLGridItem()
        self.pcplot.addItem(self.gz)
        self.pcplot.pan(dx=0, dy=1.5, dz=1.2)
        self.pcplot.setCameraPosition(distance=6.8)

        # Create scatter plot for point cloud
        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=np.zeros((1, 3)))
        self.pcplot.addItem(self.scatter)

        # Create box to represent EVM
        evmSizeX = 0.0625
        evmSizeZ = 0.125
        verts = np.empty((2, 3, 3))
        verts[0, 0, :] = [-evmSizeX, 0, evmSizeZ]
        verts[0, 1, :] = [-evmSizeX, 0, -evmSizeZ]
        verts[0, 2, :] = [evmSizeX, 0, -evmSizeZ]
        verts[1, 0, :] = [-evmSizeX, 0, evmSizeZ]
        verts[1, 1, :] = [evmSizeX, 0, evmSizeZ]
        verts[1, 2, :] = [evmSizeX, 0, -evmSizeZ]
        self.evmBox = gl.GLMeshItem(
            vertexes=verts,
            smooth=False,
            drawEdges=True,
            edgeColor=pg.glColor("r"),
            drawFaces=False,
        )
        self.pcplot.addItem(self.evmBox)

        # Initialize other elements
        self.boundaryBoxViz = []
        self.coordStr = []
        self.classifierStr = []
        self.ellipsoids = []

    def init1dGraph(self):
        self.rangePlot = pg.PlotWidget()
        self.rangePlot.setBackground("w")
        self.rangePlot.showGrid(x=True, y=True)
        self.rangePlot.setXRange(
            0, self.chirpComnCfg["NumOfAdcSamples"] / 2, padding=0.01
        )
        self.rangePlot.setYRange(0, 150, padding=0.01)
        self.rangePlot.setMouseEnabled(False, False)
        self.rangeData = pg.PlotCurveItem(pen=pg.mkPen(width=3, color="r"))
        self.rangePlot.addItem(self.rangeData)

        self.rangePlot.getPlotItem().setLabel("bottom", "Range (meters)")
        self.rangePlot.getPlotItem().setLabel("left", "Relative Power (dB)")

    def updateGraph(self, outputDict):
        pointCloud = None
        numPoints = 0
        classifierOutput = None
        tracks = None
        trackIndexs = None
        numTracks = 0
        self.frameNum = 0
        error = 0
        occupancyStates = None
        # vitalsDict = None
        # rangeProfile = None
        self.useFilter = 0
        heights = None
        enhancedPresenceDet = None
        gestureNeuralNetProb = None
        gesture = None
        # gesturePresence = None
        gestureFeatures = None
        powerData = None

        # Point Cloud
        if "pointCloud" in outputDict:
            pointCloud = outputDict["pointCloud"]

        # Number of Points
        if "numDetectedPoints" in outputDict:
            numPoints = outputDict["numDetectedPoints"]

        # Tracks
        if "trackData" in outputDict:
            tracks = outputDict["trackData"]

        # Heights
        if "heightData" in outputDict:
            heights = outputDict["heightData"]

        # Track index
        if "trackIndexes" in outputDict:
            trackIndexs = outputDict["trackIndexes"]

        # Number of Tracks
        if "numDetectedTracks" in outputDict:
            numTracks = outputDict["numDetectedTracks"]

        # Frame number
        if "frameNum" in outputDict:
            self.frameNum = outputDict["frameNum"]

        # Error
        if "error" in outputDict:
            error = outputDict["error"]

        # Range Profile
        if "rangeProfile" in outputDict:
            rangeProfile = outputDict["rangeProfile"]

        # Range Profile Major
        if "rangeProfileMajor" in outputDict:
            rangeProfileMajor = outputDict["rangeProfileMajor"]

        # Range Profile
        if "rangeProfileMinor" in outputDict:
            rangeProfileMinor = outputDict["rangeProfileMinor"]

        # Occupancy State Machine
        if "occupancy" in outputDict:
            occupancyStates = outputDict["occupancy"]

        # Enhanced Presence Detection
        if "enhancedPresenceDet" in outputDict:
            enhancedPresenceDet = outputDict["enhancedPresenceDet"]

        # Vital Signs Info
        if "vitals" in outputDict:
            vitalsDict = outputDict["vitals"]

        # Classifier Info
        if "classifierOutput" in outputDict:
            classifierOutput = outputDict["classifierOutput"]

        # Gesture neural network output probabilities
        if "gestureNeuralNetProb" in outputDict:
            gestureNeuralNetProb = outputDict["gestureNeuralNetProb"]

        # Gesture extracted features
        if "gestureFeatures" in outputDict:
            gestureFeatures = outputDict["gestureFeatures"]

        # Gesture post-processed classifier output
        if "gesture" in outputDict:
            gesture = outputDict["gesture"]

        # Gesture/presence mode flag
        if "gesturePresence" in outputDict:
            gesturePresence = outputDict["gesturePresence"]

        if "powerData" in outputDict:
            powerData = outputDict["powerData"]

        # Surface classifier output
        if "surfaceClassificationOutput" in outputDict:
            surfaceClassificationResult = outputDict["surfaceClassificationOutput"]
        else:
            surfaceClassificationResult = None
        if error != 0:
            print("Parsing Error on frame: %d" % (self.frameNum))
            print("/tError Number: %d" % (error))

        # Update text for display
        self.numPointsDisplay.setText("Points: " + str(numPoints))
        # self.numTargetsDisplay.setText("Targets: " + str(numTracks))

        # Rotate point cloud and tracks to account for elevation and azimuth tilt
        if self.profile["elev_tilt"] != 0 or self.profile["az_tilt"] != 0:
            if pointCloud is not None:
                for i in range(numPoints):
                    rotX, rotY, rotZ = eulerRot(
                        pointCloud[i, 0],
                        pointCloud[i, 1],
                        pointCloud[i, 2],
                        self.profile["elev_tilt"],
                        self.profile["az_tilt"],
                    )
                    pointCloud[i, 0] = rotX
                    pointCloud[i, 1] = rotY
                    pointCloud[i, 2] = rotZ
            if tracks is not None:
                for i in range(numTracks):
                    rotX, rotY, rotZ = eulerRot(
                        tracks[i, 1],
                        tracks[i, 2],
                        tracks[i, 3],
                        self.profile["elev_tilt"],
                        self.profile["az_tilt"],
                    )
                    tracks[i, 1] = rotX
                    tracks[i, 2] = rotY
                    tracks[i, 3] = rotZ

        # Shift points to account for sensor height
        if self.profile["sensorHeight"] != 0:
            if pointCloud is not None:
                pointCloud[:, 2] = pointCloud[:, 2] + self.profile["sensorHeight"]
            if tracks is not None:
                tracks[:, 3] = tracks[:, 3] + self.profile["sensorHeight"]

        # Update boundary box colors based on results of Occupancy State Machine
        if occupancyStates is not None:
            for box in self.boundaryBoxes:
                if "occZone" in box["name"]:
                    # Get index of the occupancy zone from the box name
                    occIdx = int(box["name"].lstrip(string.ascii_letters))
                    # Zone unnoccupied
                    if occIdx >= len(occupancyStates) or not occupancyStates[occIdx]:
                        box["color"].setCurrentText("Green")
                    # Zone occupied
                    else:
                        # Make first box turn red
                        if occIdx == 0:
                            box["color"].setCurrentText("Red")
                        else:
                            box["color"].setCurrentText("Yellow")

        # Update boundary box colors based on results of Occupancy State Machine
        if enhancedPresenceDet is not None:
            for box in self.boundaryBoxes:
                if "mpdBox" in box["name"]:
                    # Get index of the occupancy zone from the box name
                    boxIdx = int(box["name"].lstrip(string.ascii_letters))
                    # out of bounds
                    if boxIdx >= len(enhancedPresenceDet):
                        print("Warning : Occupancy results for box that does not exist")
                    elif enhancedPresenceDet[boxIdx] == 0:
                        box["color"].setCurrentText("Blue")  # Zone unoccupied
                    elif enhancedPresenceDet[boxIdx] == 1:
                        box["color"].setCurrentText(
                            "Yellow"
                        )  # Minor Motion Zone Occupancy
                    elif enhancedPresenceDet[boxIdx] == 2:
                        box["color"].setCurrentText(
                            "Red"
                        )  # Major Motion Zone Occupancy
                    else:
                        print(
                            "Error : invalid result for Enhanced Presence Detection TLV"
                        )

        # Reset all heights each loop to delete heights from tracks that disappear.
        for cstr in self.coordStr:
            cstr.setVisible(False)

        ## Visualize Target Heights
        # If fall detection is enabled

        # If there are heights to display
        t0 = threading.Thread(target=visualizePointCloud, args=(heights, tracks, self))
        t1 = threading.Thread(target=sentToFirebase, args=())
        t2 = threading.Thread(target=predictModel, args=())
        t3 = threading.Thread(target=predictFatalFall, args=())

        t0.start()
        t1.start()
        t2.start()
        t3.start()

        # Point cloud Persistence
        numPersistentFrames = int(self.persistentFramesInput.currentText())
        if self.configType.currentText() == DEMO_NAME_3DPC:
            numPersistentFrames = numPersistentFrames + 1

        # Add trackIndexs to the point cloud before adding it to the cumulative cloud
        if trackIndexs is not None:
            # Small Obstacle Detection demo doesnt support track indexes
            if (
                self.configType.currentText()
                == DEMO_NAME_3DPC
                # or self.configType.currentText() == DEMO_NAME_VITALS
            ):
                if (
                    self.previousClouds[len(self.previousClouds) - 1].shape[0]
                    != trackIndexs.shape[0]
                ):
                    # If there was no data from this frame then don't worry about this check
                    if self.lastFrameErrorFlag == False:
                        print(
                            "Warning in gui_main.py: number of points in last frame ("
                            + str(
                                self.previousClouds[len(self.previousClouds) - 1].shape[
                                    0
                                ]
                            )
                            + ") does not match number of track indexes ("
                            + str(trackIndexs.shape[0])
                            + ")"
                        )
                else:
                    self.previousClouds[len(self.previousClouds) - 1][
                        :, 6
                    ] = trackIndexs
            else:
                if pointCloud.shape[0] != trackIndexs.shape[0]:
                    print(
                        "Warning in gui_main.py: number of points does not match number of track indexes"
                    )
                else:
                    pointCloud[:, 6] = trackIndexs

        # Reset all heights each loop to delete heights from tracks that disappear.
        for cstr in self.classifierStr:
            cstr.setVisible(False)

        # Hold the track IDs detected in the current frame
        trackIDsInCurrFrame = []
        # Add classifier results with filtering to mimic MATLAB results
        if classifierOutput is not None:
            # Loop through the tracks detected to label them as human/non-human
            for trackNum, trackName in enumerate(tracks):
                # Decode trackID from the trackName
                trackID = int(trackName[0])
                # Hold the track IDs detected in the current frame
                trackIDsInCurrFrame.append(trackID)
                # Track Velocity (radial) = (x * v_x + y*v_y + z*v_z)/ r
                trackVelocity = (
                    trackName[1] * trackName[4]
                    + trackName[2] * trackName[5]
                    + trackName[3] * trackName[6]
                ) / math.sqrt(
                    math.pow(trackName[1], 2)
                    + math.pow(trackName[2], 2)
                    + math.pow(trackName[3], 2)
                )

                # Update the tags if ((classification probabilities have been generated by the radar for the current frame) AND
                # (either the target has not already been detected as a human or the doppler is above the minimum velocity for classification)).
                # This is designed to stop the tags from being assigned if target has already been detected as a human and becomes stationary.
                if classifierOutput[trackNum][0] != 0.5 and not (
                    self.wasTargetHuman[trackID] == 1
                    and abs(trackVelocity) < MIN_CLASSIFICATION_VELOCITY
                ):
                    # See if either label is above the minimum score needed for classification, it so, add the corresponding tag to the buffer
                    for label in range(NUM_CLASSES_IN_CLASSIFIER):
                        if (
                            classifierOutput[trackNum][label]
                            > CLASSIFIER_CONFIDENCE_SCORE
                        ):
                            self.classifierTags[trackID].appendleft(
                                -1 if label == 0 else 1
                            )

                ## Recompute sum of tags and number of unknown tags
                # Sum the Tags (composed of +1 for one label, -1 for the other label and 0 for unknown) to see which label is dominant
                sumOfTags = sum(self.classifierTags[trackID])
                # Count the number of times there is an unknown tag in the tag buffer
                numUnknownTags = sum(1 for i in self.classifierTags[trackID] if i == 0)

                ## Assign Labels
                # If we don't have enough tags for a decision or the number of tags for human/nonhuman are equal, make no decision
                if (
                    numUnknownTags > MAX_NUM_UNKNOWN_TAGS_FOR_HUMAN_DETECTION
                    or sumOfTags == 0
                ):
                    self.wasTargetHuman[trackID] = (
                        0  # Target was NOT detected to be human in the current frame, save for next frame
                    )
                    self.classifierStr[trackID].setText("Unknown Label")
                # If we have enough tags and the majority of them are for nonhuman, then detect nonhuman
                elif sumOfTags < 0:
                    self.wasTargetHuman[trackID] = (
                        0  # Target was NOT detected to be human in the current frame, save for next frame
                    )
                    self.classifierStr[trackID].setText("Non-Human")
                # If we have enough tags and the majority of them are for human, then detect human
                elif sumOfTags > 0:
                    self.wasTargetHuman[trackID] = (
                        1  # Target WAS detected to be human in the current frame, save for next frame
                    )
                    self.classifierStr[trackID].setText("Human")
                # Populate string that will display a label
                self.classifierStr[trackID].setX(trackName[1])
                self.classifierStr[trackID].setY(trackName[2])
                self.classifierStr[trackID].setZ(
                    trackName[3] + 0.1
                )  # Add 0.1 so it doesn't interfere with height text if enabled
                self.classifierStr[trackID].setVisible(True)

            # Regardless of whether you get tracks in the current frame, if there were tracks in the previous frame, reset the
            # tag buffer and wasHumanTarget flag for tracks that aren't detected in the current frame but were detected in the previous frame
            tracksToShuffle = set(self.tracksIDsInPreviousFrame) - set(
                trackIDsInCurrFrame
            )
            for track in tracksToShuffle:
                for frame in range(TAG_HISTORY_LEN):
                    self.classifierTags[track].appendleft(
                        0
                    )  # fill the buffer with zeros to remove any history for the track
                self.wasTargetHuman[trackID] = (
                    0  # Since target was not detected in current frame, reset the wasTargetHuman flag
                )

            # Put the current tracks detected into the previous track list for the next frame
            self.tracksIDsInPreviousFrame = copy.deepcopy(trackIDsInCurrFrame)

        # Add current point cloud to the cumulative cloud if it's not empty
        if pointCloud is not None:
            self.previousClouds.append(pointCloud)
            self.lastFrameErrorFlag == False
        else:
            self.lastFrameErrorFlag = True

        # If we have more point clouds than needed, stated by numPersistentFrames, delete the oldest ones
        while len(self.previousClouds) > numPersistentFrames:
            self.previousClouds.pop(0)

        # Since track indexes are delayed a frame on the IWR6843 demo, delay showing the current points by 1 frame
        if (self.parser.parserType == "DoubleCOMPort") and (
            self.frameNum > 1 and (self.configType.currentText() == DEMO_NAME_3DPC)
        ):
            cumulativeCloud = np.concatenate(self.previousClouds[:-1])
        elif len(self.previousClouds) > 0:
            cumulativeCloud = np.concatenate(self.previousClouds)

        # Update 3D Plot
        if self.graphTabs.currentWidget() == self.pcplot:
            # Update graph, but first ensure the last update completed
            if self.graphFin:
                self.plotstart = int(round(time.time() * 1000))
                self.graphFin = 0
                self.get_thread = updateQTTargetThread3D(
                    cumulativeCloud,
                    tracks,
                    self.scatter,
                    self.pcplot,
                    numTracks,
                    self.ellipsoids,
                    self.coordStr,
                    classifierOutput,
                    self.zRange,
                    self.colorGradient,
                    self.pointColorMode.currentText(),
                    self.plotTracks.isChecked(),
                    self.trackColorMap,
                    self.pointBounds,
                )
                self.get_thread.done.connect(self.graphDone)
                self.get_thread.start(priority=QThread.HighestPriority - 1)

        elif self.graphTabs.currentWidget() == self.rangePlot:

            # TODO add logic here to plot major or minor depending on gui monitor input
            if (
                self.parser.parserType == "DoubleCOMPort"
            ):  # Range plot not supported on 6843
                self.rangePlot.getPlotItem().setLabel("top", "range profile disabled")
            elif self.guiMonitor["rangeProfile"] == 0:
                self.rangePlot.getPlotItem().setLabel("top", "range profile disabled")
            elif self.guiMonitor["rangeProfile"] == 1:
                if rangeProfileMajor is not None:
                    self.plotstart = int(round(time.time() * 1000))
                    numRangeBinsParsed = len(rangeProfileMajor)
                    # Check size of rangeData matches expected size
                    if numRangeBinsParsed == next_power_of_2(
                        round(self.chirpComnCfg["NumOfAdcSamples"] / 2)
                    ):

                        rangeProfileMajor = np.log10(rangeProfileMajor) * 20

                        # Update graph data
                        self.rangeData.setData(self.rangeAxisVals, rangeProfileMajor)
                    else:
                        print(
                            f'Error: Size of rangeProfileMajor (${numRangeBinsParsed}) did not match the expected size (${next_power_of_2(round(self.chirpComnCfg["NumOfAdcSamples"]/2))})'
                        )
            elif self.guiMonitor["rangeProfile"] == 2:
                if rangeProfileMinor is not None:
                    self.plotstart = int(round(time.time() * 1000))
                    numRangeBinsParsed = len(rangeProfileMinor)
                    # Check size of rangeData matches expected size
                    if numRangeBinsParsed == next_power_of_2(
                        round(self.chirpComnCfg["NumOfAdcSamples"] / 2)
                    ):
                        rangeProfileMinor = np.log10(rangeProfileMinor) * 20

                        # Update graph data
                        self.rangeData.setData(self.rangeAxisVals, rangeProfileMinor)
                    else:
                        print(
                            f'Error: Size of rangeProfileMinor (${numRangeBinsParsed}) did not match the expected size (${next_power_of_2(round(self.chirpComnCfg["NumOfAdcSamples"]/2))})'
                        )
            elif self.guiMonitor["rangeProfile"] == 3:
                self.rangePlot.getPlotItem().setLabel(
                    "middle", "Major & Minor Range Profile Mode Not Supported"
                )
            else:
                self.rangePlot.getPlotItem().setLabel(
                    "middle", "INVALID gui monitor range profile input"
                )

            self.graphDone()

        elif (
            hasattr(self, "levelsensingTab")
            and self.graphTabs.currentWidget() == self.levelsensingTab
        ):

            # TODO add logic here to plot major or minor depending on gui monitor input
            if (
                self.parser.parserType == "DoubleCOMPort"
            ):  # Range plot not supported on 6843
                self.rangePlot.getPlotItem().setLabel("top", "range profile disabled")
            elif self.guiMonitor["rangeProfile"] == 0:
                self.rangePlot.getPlotItem().setLabel("top", "range profile disabled")
                self.plotstart = int(round(time.time() * 1000))
            elif self.guiMonitor["rangeProfile"] == 1:
                if rangeProfileMajor is not None:
                    self.plotstart = int(round(time.time() * 1000))
                    numRangeBinsParsed = len(rangeProfileMajor)
                    # Check size of rangeData matches expected size
                    if numRangeBinsParsed == next_power_of_2(
                        round(self.chirpComnCfg["NumOfAdcSamples"] / 2)
                    ):

                        for i in range(len(rangeProfileMajor)):
                            rangeProfileMajor[i] += 1

                        rangeProfileMajor = np.log10(rangeProfileMajor) * 20

                        # Update graph data
                        self.rangeData.setData(self.rangeAxisVals, rangeProfileMajor)

                        # Highlighting specific points
                        for i in range(len(self.rangeAxisVals)):
                            if (
                                self.Peak1 >= self.rangeAxisVals[i]
                                and self.Peak1 < self.rangeAxisVals[i + 1]
                            ):
                                highlight_peak1 = i
                            if (
                                self.Peak2 >= self.rangeAxisVals[i]
                                and self.Peak2 < self.rangeAxisVals[i + 1]
                            ):
                                highlight_peak2 = i
                            if (
                                self.Peak3 >= self.rangeAxisVals[i]
                                and self.Peak3 < self.rangeAxisVals[i + 1]
                            ):
                                highlight_peak3 = i

                        # self.HighlightPlot.clear()
                        highlight_indices = [highlight_peak1]
                        highlight_x = [self.rangeAxisVals[i] for i in highlight_indices]
                        highlight_y = [rangeProfileMajor[i] for i in highlight_indices]
                        data = [
                            {"pos": (x_val, y_val)}
                            for x_val, y_val in zip(highlight_x, highlight_y)
                        ]
                        self.HighlightPlotPeak1.setData(data)

                        # Adding labels to highlighted points
                        for i in range(len(highlight_indices)):
                            self.peakLabel1.setPos(highlight_x[i], highlight_y[i])

                        highlight_indices = [highlight_peak2]
                        highlight_x = [self.rangeAxisVals[i] for i in highlight_indices]
                        highlight_y = [rangeProfileMajor[i] for i in highlight_indices]
                        data = [
                            {"pos": (x_val, y_val)}
                            for x_val, y_val in zip(highlight_x, highlight_y)
                        ]
                        self.HighlightPlotPeak2.setData(data)

                        for i in range(len(highlight_indices)):
                            self.peakLabel2.setPos(highlight_x[i], highlight_y[i])

                        highlight_indices = [highlight_peak3]
                        highlight_x = [self.rangeAxisVals[i] for i in highlight_indices]
                        highlight_y = [rangeProfileMajor[i] for i in highlight_indices]
                        data = [
                            {"pos": (x_val, y_val)}
                            for x_val, y_val in zip(highlight_x, highlight_y)
                        ]
                        self.HighlightPlotPeak3.setData(data)

                        for i in range(len(highlight_indices)):
                            self.peakLabel3.setPos(highlight_x[i], highlight_y[i])

                    else:
                        print(
                            f'Error: Size of rangeProfileMajor (${numRangeBinsParsed}) did not match the expected size (${next_power_of_2(round(self.chirpComnCfg["NumOfAdcSamples"]/2))})'
                        )
            else:
                self.rangePlot.getPlotItem().setLabel(
                    "middle", "INVALID gui monitor range profile input"
                )

            self.updateLevelSensingPeaks()

            if powerData is not None:
                self.updateLevelSensingPower(powerData)

            self.graphDone()
        elif (
            hasattr(self, "gestureTab")
            and self.graphTabs.currentWidget() == self.gestureTab
        ):
            self.plotstart = int(round(time.time() * 1000))
            self.graphDone()
        elif (
            hasattr(self, "surfaceTab")
            and self.graphTabs.currentWidget() == self.surfaceTab
        ):
            self.plotstart = int(round(time.time() * 1000))
            self.graphDone()
        else:
            print(
                f"Warning: Invalid Widget Selected: ${self.graphTabs.currentWidget()}"
            )

    def graphDone(self):
        plotend = int(round(time.time() * 1000))
        plotime = plotend - self.plotstart
        try:
            if self.frameNum > 1:
                self.averagePlot = (plotime * 1 / self.frameNum) + (
                    self.averagePlot * (self.frameNum - 1) / (self.frameNum)
                )
            else:
                self.averagePlot = plotime
        except:
            self.averagePlot = plotime
        self.graphFin = 1
        pltstr = "Average Plot time: " + str(plotime)[:5] + " ms"
        fnstr = "Frame: " + str(self.frameNum)
        self.frameNumDisplay.setText(fnstr)
        # self.plotTimeDisplay.setText(pltstr)

    def connectCom(self):
        self.parser.frameTime = self.frameTime
        print("Parser type: ", self.configType.currentText())
        # init threads and timers
        self.uart_thread = parseUartThread(self.parser)
        if self.configType.currentText() != "Replay":
            self.uart_thread.fin.connect(self.parseData)
        self.uart_thread.fin.connect(self.updateGraph)
        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)
        try:
            uart = "COM" + self.cliCom.text()
            data = "COM" + self.dataCom.text()
            if (
                self.deviceType.currentText() in DEVICE_LIST[0:2]
            ):  # If using x843 device
                self.parser.connectComPorts(uart, data)
            self.connectStatus.setText("Connected")
        # TODO: create the disconnect button action
        except Exception as e:
            print(e)
            self.connectStatus.setText("Unable to Connect")
        if self.configType.currentText() == "Replay":
            self.connectStatus.setText("Replay")

    def updateNumTracksBuffer(self):
        # Classifier Data
        # Use a deque here because the append operation adds items to the back and pops the front
        self.classifierTags = [
            deque([0] * TAG_HISTORY_LEN, maxlen=TAG_HISTORY_LEN)
            for i in range(self.profile["maxTracks"])
        ]
        self.tracksIDsInPreviousFrame = []
        self.wasTargetHuman = [0 for i in range(self.profile["maxTracks"])]
        if self.configType.currentText() == DEMO_NAME_3DPC:
            self.fallDetection = FallDetection(self.profile["maxTracks"])

    # Select and parse the configuration file
    # Use the most recently used cfg file path as the default option
    def selectCfg(self):
        try:
            file = self.selectFile()
            cachedData.setCachedCfgPath(file)  # cache the file and demo used
            self.parseCfg(file)
            if "maxTracks" in self.profile:
                self.updateNumTracksBuffer()  # Update the max number of tracks based off the config file
        except Exception as e:
            print(e)
            print("No cfg file selected!")

    def selectFile(self):
        try:
            current_dir = os.getcwd()
            configDirectory = current_dir
            path = cachedData.getCachedCfgPath()
            if path != "":
                configDirectory = path
        except:
            configDirectory = ""

        fd = QFileDialog()
        filt = "cfg(*.cfg)"
        filename = fd.getOpenFileName(directory=configDirectory, filter=filt)
        return filename[0]

    # Add a boundary box to the boundary boxes tab
    def addBoundBox(self, name, minX=0, maxX=0, minY=0, maxY=0, minZ=0, maxZ=0):
        newBox = self.setBoxControlLayout(name)
        self.boundaryBoxes.append(newBox)
        self.boundaryBoxViz.append(gl.GLLinePlotItem())
        boxIndex = len(self.boundaryBoxes) - 1
        self.boxTab.addTab(newBox["boxCon"], name)
        self.boundaryBoxes[boxIndex]["boundList"][0].setText(str(minX))
        self.boundaryBoxes[boxIndex]["boundList"][1].setText(str(maxX))
        self.boundaryBoxes[boxIndex]["boundList"][2].setText(str(minY))
        self.boundaryBoxes[boxIndex]["boundList"][3].setText(str(maxY))
        self.boundaryBoxes[boxIndex]["boundList"][4].setText(str(minZ))
        self.boundaryBoxes[boxIndex]["boundList"][5].setText(str(maxZ))

        # Specific functionality for various types of boxes
        # Point boundary box
        if "pointBounds" in name:
            desc = "Remove points outside of this zone/nDefaults to last boundaryBox in .cfg"
            self.boundaryBoxes[boxIndex]["description"].setText(desc)
            self.boundaryBoxes[boxIndex]["checkEnable"].setDisabled(False)
        # Zone occupancy box
        elif "occZone" in name:
            desc = "Checks occupancy status on these zones"
            self.boundaryBoxes[boxIndex]["description"].setText(desc)
            self.boundaryBoxes[boxIndex]["checkEnable"].setChecked(True)
            self.boundaryBoxes[boxIndex]["color"].setCurrentText("Green")
            # Lock each text field
            for textBox in self.boundaryBoxes[boxIndex]["boundList"]:
                textBox.setDisabled(True)
            # Lock enable box
            self.boundaryBoxes[boxIndex]["checkEnable"].setDisabled(True)
            self.boundaryBoxes[boxIndex]["color"].setDisabled(True)
        elif "trackerBounds" in name:
            desc = "Checks for tracks in this zone"
            self.boundaryBoxes[boxIndex]["description"].setText(desc)
            self.boundaryBoxes[boxIndex]["checkEnable"].setChecked(True)
            # Lock each text field
            for textBox in self.boundaryBoxes[boxIndex]["boundList"]:
                textBox.setDisabled(True)
            # Lock enable box
            self.boundaryBoxes[boxIndex]["checkEnable"].setDisabled(True)
        elif "mpdBox" in name:
            desc = "checks for motion or presence in the box"
            self.boundaryBoxes[boxIndex]["description"].setText(desc)
            self.boundaryBoxes[boxIndex]["checkEnable"].setChecked(True)
            self.boundaryBoxes[boxIndex]["color"].setCurrentText("Blue")
            # Lock each text field
            for textBox in self.boundaryBoxes[boxIndex]["boundList"]:
                textBox.setDisabled(True)
            # Lock enable box
            self.boundaryBoxes[boxIndex]["checkEnable"].setDisabled(True)
            self.boundaryBoxes[boxIndex]["color"].setDisabled(True)
        # Set visible if enabled
        if self.boundaryBoxes[boxIndex]["checkEnable"].isChecked():
            self.boundaryBoxViz[boxIndex].setVisible(True)
        else:
            self.boundaryBoxViz[boxIndex].setVisible(False)
        self.pcplot.addItem(self.boundaryBoxViz[boxIndex])
        self.onChangeBoundaryBox()

    def parseCfg(self, fname):
        with open(fname, "r") as cfg_file:
            self.cfg = cfg_file.readlines()
        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if len(args) > 0:
                # cfarCfg
                if args[0] == "cfarCfg":
                    pass
                    # self.cfarConfig = {args[10], args[11], '1'}
                # trackingCfg
                elif args[0] == "trackingCfg":
                    if len(args) < 5:
                        print("Error: trackingCfg had fewer arguments than expected")
                        continue
                    self.profile["maxTracks"] = int(args[4])
                    # Update the maximum number of tracks based off the cfg file
                    self.trackColorMap = get_trackColors(self.profile["maxTracks"])
                    for m in range(self.profile["maxTracks"]):
                        # Add track gui object
                        mesh = gl.GLLinePlotItem()
                        mesh.setVisible(False)
                        self.pcplot.addItem(mesh)
                        self.ellipsoids.append(mesh)
                        # Add track coordinate string
                        text = GLTextItem()
                        text.setGLViewWidget(self.pcplot)
                        text.setVisible(False)
                        self.pcplot.addItem(text)
                        self.coordStr.append(text)
                        # Add track classifier label string
                        classifierText = GLTextItem()
                        classifierText.setGLViewWidget(self.pcplot)
                        classifierText.setVisible(False)
                        self.pcplot.addItem(classifierText)
                        self.classifierStr.append(classifierText)

                elif args[0] == "AllocationParam":
                    pass
                    # self.allocConfig = tuple(args[1:6])
                elif args[0] == "GatingParam":
                    pass
                    # self.gatingConfig = tuple(args[1:4])
                elif args[0] == "SceneryParam" or args[0] == "boundaryBox":
                    if len(args) < 7:
                        print(
                            "Error: SceneryParam/boundaryBox had fewer arguments than expected"
                        )
                        continue
                    self.boundaryLine = counter
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])
                    self.addBoundBox(
                        "trackerBounds", leftX, rightX, nearY, farY, bottomZ, topZ
                    )
                    # Default pointBounds box to have the same values as the last boundaryBox in the config
                    # These can be changed by the user
                    self.boundaryBoxes[0]["boundList"][0].setText(str(leftX))
                    self.boundaryBoxes[0]["boundList"][1].setText(str(rightX))
                    self.boundaryBoxes[0]["boundList"][2].setText(str(nearY))
                    self.boundaryBoxes[0]["boundList"][3].setText(str(farY))
                    self.boundaryBoxes[0]["boundList"][4].setText(str(bottomZ))
                    self.boundaryBoxes[0]["boundList"][5].setText(str(topZ))
                elif args[0] == "staticBoundaryBox":
                    self.staticLine = counter
                elif args[0] == "profileCfg":
                    if len(args) < 12:
                        print("Error: profileCfg had fewer arguments than expected")
                        continue
                    self.profile["startFreq"] = float(args[2])
                    self.profile["idle"] = float(args[3])
                    self.profile["adcStart"] = float(args[4])
                    self.profile["rampEnd"] = float(args[5])
                    self.profile["slope"] = float(args[8])
                    self.profile["samples"] = float(args[10])
                    self.profile["sampleRate"] = float(args[11])
                    print(self.profile)
                elif args[0] == "frameCfg":
                    if len(args) < 4:
                        print("Error: frameCfg had fewer arguments than expected")
                        continue
                    self.frameTime = float(args[5])
                    self.profile["numLoops"] = float(args[3])
                    self.profile["numTx"] = float(args[2]) + 1
                elif args[0] == "chirpCfg":
                    chirpCount += 1
                elif args[0] == "sensorPosition":
                    # sensorPosition for x843 family has 3 args
                    if self.deviceType.currentText() in DEVICE_LIST[0:2]:
                        if len(args) < 4:
                            print(
                                "Error: sensorPosition had fewer arguments than expected"
                            )
                            continue
                        self.profile["sensorHeight"] = float(args[1])
                        self.profile["az_tilt"] = float(args[2])
                        self.profile["elev_tilt"] = float(args[3])

                    # sensorPosition for x432 family has 5 args
                    if (
                        self.deviceType.currentText() in DEVICE_LIST[2]
                        or self.deviceType.currentText() in DEVICE_LIST[3]
                    ):
                        if len(args) < 6:
                            print(
                                "Error: sensorPosition had fewer arguments than expected"
                            )
                            continue
                        # xOffset and yOffset are not implemented in the python code yet.
                        self.profile["xOffset"] = float(args[1])
                        self.profile["yOffset"] = float(args[2])
                        self.profile["sensorHeight"] = float(args[3])
                        self.profile["az_tilt"] = float(args[4])
                        self.profile["elev_tilt"] = float(args[5])
                # Only used for Small Obstacle Detection
                elif args[0] == "occStateMach":
                    numZones = int(args[1])
                    if numZones > 2:
                        print(
                            "ERROR: More zones specified by cfg than are supported in this GUI"
                        )
                # Only used for Small Obstacle Detection
                elif args[0] == "zoneDef":
                    if len(args) < 8:
                        print("Error: zoneDef had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    # Offset by 3 so it is in center of screen
                    minZ = float(args[6]) + self.profile["sensorHeight"]
                    maxZ = float(args[7]) + self.profile["sensorHeight"]

                    name = "occZone" + str(zoneIdx)

                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
                elif args[0] == "mpdBoundaryBox":
                    if len(args) < 8:
                        print("Error: mpdBoundaryBox had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    minZ = float(args[6])
                    maxZ = float(args[7])
                    name = "mpdBox" + str(zoneIdx)
                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)

                elif args[0] == "chirpComnCfg":
                    if len(args) < 8:
                        print("Error: chirpComnCfg had fewer arguments than expected")
                        continue
                    try:
                        self.chirpComnCfg["DigOutputSampRate"] = int(args[1])
                        self.chirpComnCfg["DigOutputBitsSel"] = int(args[2])
                        self.chirpComnCfg["DfeFirSel"] = int(args[3])
                        self.chirpComnCfg["NumOfAdcSamples"] = int(args[4])
                        self.chirpComnCfg["ChirpTxMimoPatSel"] = int(args[5])
                        self.chirpComnCfg["ChirpRampEndTime"] = 10 * float(args[6])
                        self.chirpComnCfg["ChirpRxHpfSel"] = int(args[7])
                    except Exception as e:
                        print(e)

                elif args[0] == "chirpTimingCfg":
                    if len(args) < 6:
                        print("Error: chirpTimingCfg had fewer arguments than expected")
                        continue
                    self.chirpTimingCfg["ChirpIdleTime"] = 10.0 * float(args[1])
                    self.chirpTimingCfg["ChirpAdcSkipSamples"] = int(args[2]) << 10
                    self.chirpTimingCfg["ChirpTxStartTime"] = 10.0 * float(args[3])
                    self.chirpTimingCfg["ChirpRfFreqSlope"] = float(args[4])
                    self.chirpTimingCfg["ChirpRfFreqStart"] = float(args[5])
                elif args[0] == "clusterCfg":
                    if len(args) < 4:
                        print("Error: clusterCfg had fewer arguments than expected")
                        continue
                    self.profile["enabled"] = float(args[1])
                    self.profile["maxDistance"] = float(args[2])
                    self.profile["minPoints"] = float(args[3])

                # This is specifically guiMonitor for 60Lo, this parsing will break the gui when an SDK 3 config is sent
                elif args[0] == "guiMonitor":
                    if (
                        self.deviceType.currentText() in DEVICE_LIST[2]
                        or self.deviceType.currentText() in DEVICE_LIST[3]
                    ):
                        if len(args) < 12:
                            print("Error: guiMonitor had fewer arguments than expected")
                            continue
                    self.guiMonitor["pointCloud"] = int(args[1])
                    self.guiMonitor["rangeProfile"] = int(args[2])
                    self.guiMonitor["NoiseProfile"] = int(args[3])
                    self.guiMonitor["rangeAzimuthHeatMap"] = int(args[4])
                    self.guiMonitor["rangeDopplerHeatMap"] = int(args[5])
                    self.guiMonitor["statsInfo"] = int(args[6])

            counter += 1

        # self.rangeRes = (3e8*(100/self.chirpComnCfg['DigOutputSampRate']))/(2*self.chirpTimingCfg['ChirpRfFreqSlope']*self.chirpComnCfg['NumOfAdcSamples'])
        self.rangeRes = (3e8 * (100 / self.chirpComnCfg["DigOutputSampRate"]) * 1e6) / (
            2
            * self.chirpTimingCfg["ChirpRfFreqSlope"]
            * 1e12
            * self.chirpComnCfg["NumOfAdcSamples"]
        )
        self.rangePlot.setXRange(
            0, (self.chirpComnCfg["NumOfAdcSamples"] / 2) * self.rangeRes, padding=0.01
        )

        self.rangeAxisVals = np.arange(
            0, self.chirpComnCfg["NumOfAdcSamples"] / 2 * self.rangeRes, self.rangeRes
        )
        print(self.guiMonitor["rangeProfile"])

        if self.guiMonitor["rangeProfile"] == 0:
            self.rangePlot.getPlotItem().setLabel("top", "range profile disabled")
        elif self.guiMonitor["rangeProfile"] == 1:
            self.rangePlot.getPlotItem().setLabel("top", "Major Range Profile")
        elif self.guiMonitor["rangeProfile"] == 2:
            self.rangePlot.getPlotItem().setLabel("top", "Minor Range Profile")
        elif self.guiMonitor["rangeProfile"] == 3:
            self.rangePlot.getPlotItem().setLabel(
                "top", "Major & Minor Range Profile Mode Not Supported"
            )
        else:
            self.rangePlot.getPlotItem().setLabel(
                "top", "INVALID gui monitor range profile input"
            )

        # Update sensor position
        self.az_tilt.setText(str(self.profile["az_tilt"]))
        self.elev_tilt.setText(str(self.profile["elev_tilt"]))
        self.s_height.setText(str(self.profile["sensorHeight"]))
        self.onChangeSensorPosition()

    def sendCfg(self):
        try:
            self.saveBinaryBox.setDisabled(True)
            if self.configType.currentText() != "Replay":
                self.parser.sendCfg(self.cfg)
                self.configSent = 1
                self.parseTimer.start(self.frameTime)  # need this line

        except Exception as e:
            print(e)
            print("No cfg file selected!")

    def startApp(self):
        self.configSent = 1
        self.parseTimer.start(self.frameTime)  # need this line

    def parseData(self):
        self.uart_thread.start(priority=QThread.HighestPriority)

    def whoVisible(self):
        if self.threeD:
            self.threeD = 0
        else:
            self.threeD = 1


if __name__ == "__main__":
    if compileGui:
        appctxt = ApplicationContext()
        app = QApplication(sys.argv)
        screen = app.primaryScreen()
        size = screen.size()
        main = Window(size=size)
        main.showMaximized()
        exit_code = appctxt.app.exec_()
        sys.exit(exit_code)
    else:
        QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(sys.argv)
        screen = app.primaryScreen()
        size = screen.size()
        main = Window(size=size)
        main.showMaximized()
        sys.exit(app.exec_())
