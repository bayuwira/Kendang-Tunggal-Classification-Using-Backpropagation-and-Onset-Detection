# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KendangApp2.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import pickle
import librosa, os, time, numpy as np, glob
from pydub import AudioSegment
from function.onsetValidation import onsetDetection, onsetDetectionNonNormalize, \
    plotingWave, convertTimesToSecond
from function.dataTrainFeatureDataFrame import DataTrainMaker
from function.dataTestFeatureDataFrame import DataTestMaker
from function import Backpropagation
from function.MLSA import synthesis_wav
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
import logging

log = logging

# variable global
filename = ""
onset_times_backtrack = np.array([])
normalization = True
network = list()
predicted_notes = list()
n_epoch = 0
n_lrate = 0
n_hidden = 0
k_fold = ""
progress_bar = 0
onset_sample = np.array([])
onset_times = np.array([])
class_predict = list()


class MakeWav(QtCore.QThread):
    countChanged = QtCore.pyqtSignal(int)

    def run(self):
        global filename, onset_times_backtrack
        count = 0
        onset_times_in_milliseconds = convertTimesToSecond(onset_times_backtrack)
        if not os.path.isdir("split_audio"):
            os.mkdir("split_audio")
        check_data = [files for files in glob.glob('split_audio/*')]
        if (check_data):
            for file in check_data:
                os.remove(file)
        elif (not check_data):
            pass
        audio = AudioSegment.from_file(filename)
        y, sr = librosa.load(filename)
        end_of_sample = ((1 / sr) * len(y) * 1000)
        onset_times_in_milliseconds = np.append(onset_times_in_milliseconds, end_of_sample)
        start = onset_times_in_milliseconds[0]
        # In Milliseconds, this will cut 10 Sec of audio
        # (1 Sec = 1000 milliseconds)
        len_data = len(onset_times_backtrack)
        index = 1
        for threshold in onset_times_in_milliseconds:
            count += 1
            self.countChanged.emit(int(count / len_data * 100))
            if start == threshold:
                pass
            else:
                end = threshold
                # print(start, end)
                counter = threshold
                chunk = audio[start:end]
                temporary_segment = 'split_audio/{}-potongan-{}.wav'.format(index, counter)
                chunk.export(temporary_segment, format="wav")
                start = end
                index += 1
                time.sleep(.5)


class BackpropagationNN(QtCore.QThread):
    countChanged = QtCore.pyqtSignal(int)

    def run(self):
        global n_epoch, n_lrate, n_hidden, network, progress_bar
        count = 0
        progress_bar = 0
        filename = 'data_train.csv'
        dataset = Backpropagation.load_csv(filename)
        for i in range(len(dataset[0]) - 1):
            Backpropagation.str_column_to_float(dataset, i)
        # convert class column to integers
        Backpropagation.str_column_to_int(dataset, len(dataset[0]) - 1)
        n_inputs = len(dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in dataset]))
        network = Backpropagation.initialize_network(n_inputs, n_hidden, n_outputs)
        n_datatrain = len(dataset)
        print("network sebelom", network)
        for epoch in range(n_epoch):
            sum_error = 0
            for row in dataset:
                outputs = Backpropagation.forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                Backpropagation.backward_propagate_error(network, expected)
                Backpropagation.update_weights(network, row, n_lrate)
            mse = sum_error / n_datatrain
            count += 1
            progress_bar = int(count / n_epoch * 100)
            self.countChanged.emit(int(count / n_epoch * 100))
            print("=> epoch = %d, lrate = %.2f, error= %.5f" % (epoch, n_lrate, mse))

        print("network sesudah", network)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 800)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/kendang.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 950, 800))
        font = QtGui.QFont()
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.tabWidget.setFont(font)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.get_onset_btn = QtWidgets.QPushButton(self.tab)
        self.get_onset_btn.setGeometry(QtCore.QRect(50, 80, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setBold(False)
        font.setWeight(50)
        self.get_onset_btn.setFont(font)
        self.get_onset_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.get_onset_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/audio-waves.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.get_onset_btn.setIcon(icon)
        self.get_onset_btn.setIconSize(QtCore.QSize(50, 50))
        self.get_onset_btn.setObjectName("get_onset_btn")
        self.split_notes_btn = QtWidgets.QPushButton(self.tab)
        self.split_notes_btn.setEnabled(False)
        self.split_notes_btn.setGeometry(QtCore.QRect(400, 80, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setBold(False)
        font.setWeight(50)
        self.split_notes_btn.setFont(font)
        self.split_notes_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.split_notes_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/wave.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.split_notes_btn.setIcon(icon1)
        self.split_notes_btn.setIconSize(QtCore.QSize(50, 50))
        self.split_notes_btn.setObjectName("split_notes_btn")
        self.notes_feature_extract = QtWidgets.QPushButton(self.tab)
        self.notes_feature_extract.setEnabled(False)
        self.notes_feature_extract.setGeometry(QtCore.QRect(750, 80, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(50)
        self.notes_feature_extract.setFont(font)
        self.notes_feature_extract.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.notes_feature_extract.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/search_color.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.notes_feature_extract.setIcon(icon2)
        self.notes_feature_extract.setIconSize(QtCore.QSize(50, 50))
        self.notes_feature_extract.setObjectName("notes_feature_extract")
        self.onset_image_container = QtWidgets.QLabel(self.tab)
        self.onset_image_container.setGeometry(QtCore.QRect(50, 260, 550, 250))
        font = QtGui.QFont()
        font.setFamily("MT Extra")
        self.onset_image_container.setFont(font)
        self.onset_image_container.setFrameShape(QtWidgets.QFrame.Box)
        self.onset_image_container.setText("")
        self.onset_image_container.setPixmap(QtGui.QPixmap("figure/placeholder.png"))
        self.onset_image_container.setScaledContents(True)
        self.onset_image_container.setAlignment(QtCore.Qt.AlignCenter)
        self.onset_image_container.setObjectName("onset_image_container")
        self.predict_btn = QtWidgets.QPushButton(self.tab)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setGeometry(QtCore.QRect(50, 560, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(50)
        self.predict_btn.setFont(font)
        self.predict_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.predict_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon/clairaudience.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.predict_btn.setIcon(icon3)
        self.predict_btn.setIconSize(QtCore.QSize(50, 50))
        self.predict_btn.setObjectName("predict_btn")
        self.sintesis_btn = QtWidgets.QPushButton(self.tab)
        self.sintesis_btn.setEnabled(False)
        self.sintesis_btn.setGeometry(QtCore.QRect(400, 560, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(50)
        self.sintesis_btn.setFont(font)
        self.sintesis_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.sintesis_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icon/music.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sintesis_btn.setIcon(icon4)
        self.sintesis_btn.setIconSize(QtCore.QSize(50, 50))
        self.sintesis_btn.setObjectName("sintesis_btn")
        self.play_btn_2 = QtWidgets.QPushButton(self.tab)
        self.play_btn_2.setEnabled(False)
        self.play_btn_2.setGeometry(QtCore.QRect(750, 560, 160, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(50)
        self.play_btn_2.setFont(font)
        self.play_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.play_btn_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icon/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play_btn_2.setIcon(icon5)
        self.play_btn_2.setIconSize(QtCore.QSize(50, 50))
        self.play_btn_2.setObjectName("play_btn_2")
        self.predict_notes_label = QtWidgets.QLabel(self.tab)
        self.predict_notes_label.setGeometry(QtCore.QRect(610, 260, 301, 141))
        self.predict_notes_label.setFrameShape(QtWidgets.QFrame.Box)
        self.predict_notes_label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.predict_notes_label.setTextFormat(QtCore.Qt.AutoText)
        self.predict_notes_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.predict_notes_label.setWordWrap(True)
        self.predict_notes_label.setIndent(-1)
        self.predict_notes_label.setObjectName("predict_notes_label")
        self.ket_4 = QtWidgets.QLabel(self.tab)
        self.ket_4.setGeometry(QtCore.QRect(610, 490, 81, 16))
        self.ket_4.setObjectName("ket_4")
        self.ket = QtWidgets.QLabel(self.tab)
        self.ket.setGeometry(QtCore.QRect(610, 420, 91, 16))
        self.ket.setObjectName("ket")
        self.ket_6 = QtWidgets.QLabel(self.tab)
        self.ket_6.setGeometry(QtCore.QRect(730, 470, 81, 16))
        self.ket_6.setObjectName("ket_6")
        self.ket_7 = QtWidgets.QLabel(self.tab)
        self.ket_7.setGeometry(QtCore.QRect(730, 490, 81, 16))
        self.ket_7.setObjectName("ket_7")
        self.ket_2 = QtWidgets.QLabel(self.tab)
        self.ket_2.setGeometry(QtCore.QRect(610, 450, 81, 16))
        self.ket_2.setObjectName("ket_2")
        self.ket_3 = QtWidgets.QLabel(self.tab)
        self.ket_3.setGeometry(QtCore.QRect(610, 470, 81, 16))
        self.ket_3.setObjectName("ket_3")
        self.ket_5 = QtWidgets.QLabel(self.tab)
        self.ket_5.setGeometry(QtCore.QRect(730, 450, 81, 16))
        self.ket_5.setObjectName("ket_5")
        self.normalizationCBK = QtWidgets.QCheckBox(self.tab)
        self.normalizationCBK.setEnabled(True)
        self.normalizationCBK.setGeometry(QtCore.QRect(220, 80, 111, 21))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        self.normalizationCBK.setFont(font)
        self.normalizationCBK.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.normalizationCBK.setAutoFillBackground(False)
        self.normalizationCBK.setChecked(True)
        self.normalizationCBK.setObjectName("normalizationCBK")
        self.normalizationCBK.stateChanged.connect(self.checkBoxChangedAction)
        self.hop_size_input = QtWidgets.QComboBox(self.tab)
        self.hop_size_input.setGeometry(QtCore.QRect(220, 150, 73, 22))
        self.hop_size_input.setEditable(True)
        self.hop_size_input.setObjectName("hop_size_input")
        self.hop_size_input.addItem("")
        self.hop_size_input.addItem("")
        self.hop_size_input.addItem("")
        self.hop_size_input.addItem("")
        self.hop_size = QtWidgets.QLabel(self.tab)
        self.hop_size.setGeometry(QtCore.QRect(220, 130, 61, 16))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        self.hop_size.setFont(font)
        self.hop_size.setObjectName("hop_size")
        self.split_notes_pg = QtWidgets.QProgressBar(self.tab)
        self.split_notes_pg.setGeometry(QtCore.QRect(400, 190, 161, 23))
        font = QtGui.QFont()
        font.setFamily("Muli Black")
        self.split_notes_pg.setFont(font)
        self.split_notes_pg.setProperty("value", 0)
        self.split_notes_pg.setObjectName("split_notes_pg")
        self.data_test_pg = QtWidgets.QProgressBar(self.tab)
        self.data_test_pg.setGeometry(QtCore.QRect(750, 190, 161, 23))
        font = QtGui.QFont()
        font.setFamily("Muli Black")
        self.data_test_pg.setFont(font)
        self.data_test_pg.setProperty("value", 0)
        self.data_test_pg.setTextVisible(True)
        self.data_test_pg.setOrientation(QtCore.Qt.Horizontal)
        self.data_test_pg.setObjectName("data_test_pg")
        self.onset_filename_label = QtWidgets.QLineEdit(self.tab)
        self.onset_filename_label.setGeometry(QtCore.QRect(49, 20, 861, 31))
        self.onset_filename_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.onset_filename_label.setReadOnly(True)
        self.onset_filename_label.setObjectName("onset_filename_label")
        self.onset_filename_label.setFont(font)
        self.title_image = QtWidgets.QLabel(self.tab)
        self.title_image.setGeometry(QtCore.QRect(50, 215, 171, 31))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(11)
        self.title_image.setFont(font)
        self.title_image.setText("")
        self.title_image.setObjectName("title_image")
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.max_epoch_input = QtWidgets.QComboBox(self.tab_3)
        self.max_epoch_input.setGeometry(QtCore.QRect(270, 120, 81, 41))
        self.max_epoch_input.setEditable(True)
        self.max_epoch_input.setObjectName("max_epoch_input")
        self.max_epoch_input.addItem("")
        self.train_btn = QtWidgets.QPushButton(self.tab_3)
        self.train_btn.setGeometry(QtCore.QRect(390, 260, 180, 100))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        font.setBold(False)
        font.setWeight(50)
        self.train_btn.setFont(font)
        self.train_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.train_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icon/athlete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.train_btn.setIcon(icon6)
        self.train_btn.setIconSize(QtCore.QSize(50, 50))
        self.train_btn.setObjectName("train_btn")
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setGeometry(QtCore.QRect(270, 30, 481, 51))
        font = QtGui.QFont()
        font.setFamily("Montserrat Black")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.learning_rate_input = QtWidgets.QComboBox(self.tab_3)
        self.learning_rate_input.setGeometry(QtCore.QRect(440, 120, 81, 41))
        self.learning_rate_input.setEditable(True)
        self.learning_rate_input.setObjectName("learning_rate_input")
        self.learning_rate_input.addItem("")
        self.learning_rate_input.addItem("")
        self.learning_rate_input.addItem("")
        self.hidden_layer_input = QtWidgets.QComboBox(self.tab_3)
        self.hidden_layer_input.setGeometry(QtCore.QRect(610, 120, 81, 41))
        self.hidden_layer_input.setEditable(True)
        self.hidden_layer_input.setObjectName("hidden_layer_input")
        self.hidden_layer_input.addItem("")
        self.hidden_layer_input.addItem("")
        self.hidden_layer_input.addItem("")
        self.train_pbar = QtWidgets.QProgressBar(self.tab_3)
        self.train_pbar.setGeometry(QtCore.QRect(260, 410, 461, 23))
        font = QtGui.QFont()
        font.setFamily("Muli Black")
        self.train_pbar.setFont(font)
        self.train_pbar.setProperty("value", 0)
        self.train_pbar.setObjectName("train_pbar")
        self.max_epoch = QtWidgets.QLabel(self.tab_3)
        self.max_epoch.setGeometry(QtCore.QRect(260, 170, 100, 30))
        self.max_epoch.setAlignment(QtCore.Qt.AlignCenter)
        self.max_epoch.setObjectName("max_epoch")
        self.learning_rate = QtWidgets.QLabel(self.tab_3)
        self.learning_rate.setGeometry(QtCore.QRect(430, 170, 100, 30))
        self.learning_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.learning_rate.setObjectName("learning_rate")
        self.hidden_layer = QtWidgets.QLabel(self.tab_3)
        self.hidden_layer.setGeometry(QtCore.QRect(600, 170, 100, 30))
        self.hidden_layer.setAlignment(QtCore.Qt.AlignCenter)
        self.hidden_layer.setObjectName("hidden_layer")
        self.save_model_btn = QtWidgets.QPushButton(self.tab_3)
        self.save_model_btn.setEnabled(True)
        self.save_model_btn.setGeometry(QtCore.QRect(390, 640, 180, 80))
        font = QtGui.QFont()
        font.setFamily("Montserrat Light")
        self.save_model_btn.setFont(font)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icon/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_model_btn.setIcon(icon7)
        self.save_model_btn.setIconSize(QtCore.QSize(50, 50))
        self.save_model_btn.setObjectName("save_model_btn")
        self.img_backpro = QtWidgets.QLabel(self.tab_3)
        self.img_backpro.setGeometry(QtCore.QRect(320, 450, 321, 181))
        self.img_backpro.setText("")
        self.img_backpro.setPixmap(QtGui.QPixmap("asset/img_backpro.jpeg"))
        self.img_backpro.setScaledContents(True)
        self.img_backpro.setObjectName("img_backpro")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.data_frame_btn = QtWidgets.QPushButton(self.tab_2)
        self.data_frame_btn.setGeometry(QtCore.QRect(410, 230, 160, 100))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icon/csv.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.data_frame_btn.setIcon(icon8)
        self.data_frame_btn.setIconSize(QtCore.QSize(50, 50))
        self.data_frame_btn.setCheckable(False)
        self.data_frame_btn.setChecked(False)
        self.data_frame_btn.setObjectName("data_frame_btn")
        self.df_pbar = QtWidgets.QProgressBar(self.tab_2)
        self.df_pbar.setGeometry(QtCore.QRect(350, 420, 291, 23))
        self.df_pbar.setProperty("value", 0)
        self.df_pbar.setOrientation(QtCore.Qt.Horizontal)
        self.df_pbar.setInvertedAppearance(False)
        self.df_pbar.setObjectName("df_pbar")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(350, 90, 291, 51))
        font = QtGui.QFont()
        font.setFamily("Montserrat Black")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.data_saved_label = QtWidgets.QLabel(self.tab_2)
        self.data_saved_label.setGeometry(QtCore.QRect(350, 460, 281, 31))
        font = QtGui.QFont()
        font.setFamily("MS PGothic")
        font.setPointSize(11)
        self.data_saved_label.setFont(font)
        self.data_saved_label.setTextFormat(QtCore.Qt.AutoText)
        self.data_saved_label.setScaledContents(False)
        self.data_saved_label.setWordWrap(True)
        self.data_saved_label.setObjectName("data_saved_label")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 950, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.onset_openfile_btn = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icon/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.onset_openfile_btn.setIcon(icon9)
        self.onset_openfile_btn.setObjectName("onset_openfile_btn")
        self.actionQuit_2 = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("icon/remove.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionQuit_2.setIcon(icon10)
        self.actionQuit_2.setObjectName("actionQuit_2")
        self.actionPlay_File = QtWidgets.QAction(MainWindow)
        self.actionPlay_File.setCheckable(False)
        self.actionPlay_File.setEnabled(False)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("icon/edit_play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPlay_File.setIcon(icon11)
        self.actionPlay_File.setObjectName("actionPlay_File")
        self.menuFile.addAction(self.onset_openfile_btn)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit_2)
        self.menuEdit.addAction(self.actionPlay_File)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ''' all button function start Here'''
        self.onset_openfile_btn.triggered.connect(self.open_file_dialog)
        self.actionPlay_File.triggered.connect(self.play_file)
        self.actionQuit_2.triggered.connect(QtWidgets.QApplication.instance().quit)
        self.get_onset_btn.clicked.connect(self.get_onset)
        self.split_notes_btn.clicked.connect(self.split_notes)
        self.data_frame_btn.clicked.connect(self.make_data_frame)
        self.notes_feature_extract.clicked.connect(self.make_data_test)
        self.train_btn.clicked.connect(self.train_backpro)
        self.save_model_btn.clicked.connect(self.save_model)
        self.predict_btn.clicked.connect(self.predict_notes)
        self.sintesis_btn.clicked.connect(self.sintesis_audio)
        self.play_btn_2.clicked.connect(self.play_audio)

    def play_file(self):
        global filename
        self.url = QtCore.QUrl.fromLocalFile(filename)
        self.content = QtMultimedia.QMediaContent(self.url)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(self.content)
        self.player.play()

    def play_audio(self):
        self.url = QtCore.QUrl.fromLocalFile('audio_sintetik/synthesized_audio.wav')
        self.content = QtMultimedia.QMediaContent(self.url)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(self.content)
        self.player.play()

    def sintesis_audio(self):
        global filename, onset_sample, class_predict
        if (not filename):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("null variable")
            msg.setText("Set the parameter !")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
        else:
            try:
                y, sr = librosa.load(filename)
                print("y = {}, sr = {}, filename={}".format(y, sr, filename))
                synthesis_wav(y, sr, onset_sample, class_predict)
                self.onset_image_container.setPixmap(QtGui.QPixmap("figure/synthesized_audio.png"))
                self.play_btn_2.setEnabled(True)
                self.title_image.setText("Hasil Sintesis : ")
            except:
                log.exception("error detected")

    def predict_notes(self):
        self.predict_notes_label.clear()
        global predicted_notes, class_predict, onset_times
        filename_test = 'data_test.csv'
        dataset_test = Backpropagation.load_csv(filename_test)
        for i in range(len(dataset_test[0]) - 1):
            Backpropagation.str_column_to_float(dataset_test, i)

        # convert class column to integers
        Backpropagation.str_column_to_int(dataset_test, len(dataset_test[0]) - 1)
        with open('weight_train.pkl', 'rb') as picklefile:
            network = pickle.load(picklefile)
        answer = []
        notes = ""
        new_notes = ""
        for row in dataset_test:
            prediction = Backpropagation.predict(network, row)
            answer.append(prediction + 1)

            if (prediction + 1 == 1):
                predicted_notes.append('C')
            elif (prediction + 1 == 2):
                predicted_notes.append('D')
            elif (prediction + 1 == 3):
                predicted_notes.append('T')
            elif (prediction + 1 == 4):
                predicted_notes.append('p')
            elif (prediction + 1 == 5):
                predicted_notes.append('u')
            elif (prediction + 1 == 6):
                predicted_notes.append('t')
            else:
                predicted_notes.append('^')
        print(answer)
        diff_all = []
        for index, data in enumerate(onset_times):
            try:
                if (index + 1 < len(onset_times) or index - 1 >= 0):
                    diff = abs(onset_times[index] - onset_times[index + 1])
                    diff_all.append(diff)
            except:
                continue
        average_times = sum(diff_all) / len(diff_all)

        class_predict = answer
        for note in predicted_notes:
            notes += note + ' ?? '

        # onset_times = onset_times.tolist()
        # print((len(predicted_notes)))
        # print(len(onset_times))

        threshold = average_times + 0.1
        print("threshold : ", threshold)

        a_list = predicted_notes
        b_list = onset_times.tolist()
        for index, elem in enumerate(a_list):
            if (index + 1 < len(a_list) and index - 1 >= 0):
                diff = abs(b_list[index] - b_list[index - 1])
                print(diff)
                if (diff >= threshold):
                    for i in range(0, int(diff + 1)):
                        new_notes += ' ?? '
                new_notes += (a_list[index] + ' ?? ')
            else:
                new_notes += a_list[index]
                if (index + 1 < len(a_list)):
                    new_notes += ' ?? '

        print(new_notes)
        self.predict_notes_label.setText(new_notes)
        self.sintesis_btn.setEnabled(True)
        predicted_notes.clear()

    def save_model(self):
        global network
        with open('weight_train.pkl', 'wb') as picklefile:
            pickle.dump(network, picklefile)
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Model Alert")
        msg.setText("Model Saved as weight_train.pkl")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()

    def checkBoxChangedAction(self, state):
        global normalization
        if (QtCore.Qt.Checked == state):
            normalization = True
        else:
            normalization = False
        print(len(onset_times_backtrack))

    def open_file_dialog(self):
        global filename
        self.openFileNamesDialog()
        display_filename = filename.split("/")[-1]
        self.onset_filename_label.setText(display_filename)
        if (filename):
            self.actionPlay_File.setEnabled(True)

    def openFileNamesDialog(self):
        global filename
        path = QtWidgets.QFileDialog.getOpenFileName(directory='dataset/data_val/')
        filename = path[0]

    def get_onset(self):
        global filename, normalization, onset_times_backtrack, onset_sample, onset_times
        hop_size = self.hop_size_input.currentText()
        if (not hop_size or not filename):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("null variable")
            msg.setText("Set the parameter !")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
        else:
            x, sr = librosa.load(filename)
            hop_size = int(hop_size)
            if (normalization):
                onset_times_backtrack, onset_sample, onset_times = onsetDetection(x, sr, hop_size)
            else:
                onset_times_backtrack, onset_sample, onset_times = onsetDetectionNonNormalize(x, sr, hop_size)
            plotingWave(x, sr, filename, hop_size, onset_times_backtrack, normalize=normalization)
            self.onset_image_container.setPixmap(QtGui.QPixmap("figure/onset.png"))
            self.split_notes_btn.setEnabled(True)
            self.title_image.setText("Hasil Onset : ")
            self.data_test_pg.setProperty("value", 0)
        print("onset_sample {}, panjang {}".format(onset_sample, len(onset_sample)))

    def split_notes(self):
        try:
            self.calc = MakeWav()
            self.calc.countChanged.connect(self.countSplitNotes)
            self.calc.start()
            self.notes_feature_extract.setEnabled(True)
        except:
            log.exception("error detected")

    def make_data_frame(self):
        self.calc = DataTrainMaker()
        self.calc.countChanged.connect(self.countDataFrame)
        self.calc.start()
        self.data_saved_label.setText("data train saved at : data_train.csv")

    def make_data_test(self):
        self.calc = DataTestMaker()
        self.calc.countChanged.connect(self.countDataTest)
        self.calc.start()
        self.predict_btn.setEnabled(True)
        self.split_notes_pg.setProperty("value", 0)

    def train_backpro(self):
        global n_epoch, n_lrate, n_hidden
        n_epoch = self.max_epoch_input.currentText()
        n_lrate = self.learning_rate_input.currentText()
        n_hidden = self.hidden_layer_input.currentText()

        if (not n_epoch or not n_lrate or not n_hidden):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("null variable")
            msg.setText("Set the parameter !")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
        else:
            n_epoch = int(n_epoch)
            n_lrate = float(n_lrate)
            n_hidden = int(n_hidden)
            self.calc = BackpropagationNN()
            self.calc.countChanged.connect(self.countTrain)
            self.calc.start()
            self.save_model_btn.setEnabled(True)

    def countSplitNotes(self, value):
        self.split_notes_pg.setValue(value)

    def countTrain(self, value):
        self.train_pbar.setValue(value)

    def countDataFrame(self, value):
        self.df_pbar.setValue(value)

    def countDataTest(self, value):
        self.data_test_pg.setValue(value)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KendangApp"))
        self.get_onset_btn.setText(_translate("MainWindow", "1. GET ONSET"))
        self.split_notes_btn.setText(_translate("MainWindow", "2. SPLIT NOTES"))
        self.notes_feature_extract.setText(_translate("MainWindow", "3. GET FEATURE"))
        self.predict_btn.setText(_translate("MainWindow", "4. PREDICT NOTES"))
        self.sintesis_btn.setText(_translate("MainWindow", "5. SYNTHESIS"))
        self.play_btn_2.setText(_translate("MainWindow", "6. PLAY PUPUH"))
        self.predict_notes_label.setText(_translate("MainWindow", "notes notation"))
        self.ket_4.setText(_translate("MainWindow", "T - Tek Kanan"))
        self.ket.setText(_translate("MainWindow", "Notes Symbol : "))
        self.ket_6.setText(_translate("MainWindow", "u - Pung Kiri"))
        self.ket_7.setText(_translate("MainWindow", "t - Teng Kiri"))
        self.ket_2.setText(_translate("MainWindow", "C - Cung Kanan"))
        self.ket_3.setText(_translate("MainWindow", "D - De Kanan"))
        self.ket_5.setText(_translate("MainWindow", "p - Pak Kiri"))
        self.normalizationCBK.setText(_translate("MainWindow", "Normalization"))
        self.hop_size_input.setCurrentText(_translate("MainWindow", "110"))
        self.hop_size_input.setItemText(0, _translate("MainWindow", "110"))
        self.hop_size_input.setItemText(1, _translate("MainWindow", "220"))
        self.hop_size_input.setItemText(2, _translate("MainWindow", "440"))
        self.hop_size_input.setItemText(3, _translate("MainWindow", "512"))
        self.hop_size.setText(_translate("MainWindow", "Hop Size"))
        self.onset_filename_label.setText(_translate("MainWindow", "<filename>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Pengujian"))
        self.max_epoch_input.setCurrentText(_translate("MainWindow", "2000"))
        self.max_epoch_input.setItemText(0, _translate("MainWindow", "2000"))
        self.train_btn.setText(_translate("MainWindow", "TRAIN NETWORK"))
        self.label.setText(_translate("MainWindow", "SET PARAMETER BACKPROPAGATION"))
        self.learning_rate_input.setCurrentText(_translate("MainWindow", "0.9"))
        self.learning_rate_input.setItemText(0, _translate("MainWindow", "0.9"))
        self.learning_rate_input.setItemText(1, _translate("MainWindow", "0.1"))
        self.learning_rate_input.setItemText(2, _translate("MainWindow", "0.5"))
        self.hidden_layer_input.setCurrentText(_translate("MainWindow", "10"))
        self.hidden_layer_input.setItemText(0, _translate("MainWindow", "10"))
        self.hidden_layer_input.setItemText(1, _translate("MainWindow", "5"))
        self.hidden_layer_input.setItemText(2, _translate("MainWindow", "15"))
        self.max_epoch.setText(_translate("MainWindow", "max epoch"))
        self.learning_rate.setText(_translate("MainWindow", "learning rate"))
        self.hidden_layer.setText(_translate("MainWindow", "hidden layer"))
        self.save_model_btn.setText(_translate("MainWindow", "save model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Klasifikasi"))
        self.data_frame_btn.setText(_translate("MainWindow", "make data frame"))
        self.label_2.setText(_translate("MainWindow", "MAKE DATA TRAINING"))
        self.data_saved_label.setText(_translate("MainWindow", "data train saved at :"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Ekstraksi Fitur"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.onset_openfile_btn.setText(_translate("MainWindow", "Open"))
        self.actionQuit_2.setText(_translate("MainWindow", "Quit"))
        self.actionPlay_File.setText(_translate("MainWindow", "Play File "))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    file = open('theme/Ubuntu.qss', 'r')
    with file:
        qss = file.read()
        app.setStyleSheet(qss)
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
