# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KendangApp.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# import-importan
import pickle
import librosa, os, time, numpy as np, glob
from pydub import AudioSegment
from onsetValidation import onsetDetection, onsetDetectionNonNormalize, \
    plotingWave, convertTimesToSecond
from dataTrainFeatureDataFrame import DataTrainMaker
from dataTestFeatureDataFrame import DataTestMaker
import Backpropagation
from MLSA import mlsa, synthesis_wav
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from qt_material import apply_stylesheet
import logging

log = logging

# variable global
filename = ""
onset_times_backtrack = np.array([])
normalization = False
network = list()
predicted_notes = list()
n_epoch = 0
n_lrate = 0
n_hidden = 0
k_fold = ""
progress_bar = 0
onset_sample = np.array([])
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
        MainWindow.resize(800, 744)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/kendang.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 801, 731))
        self.tabWidget.setMouseTracking(False)
        self.tabWidget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.onset_detection_label = QtWidgets.QLabel(self.tab_1)
        self.onset_detection_label.setGeometry(QtCore.QRect(280, 10, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Lato")
        font.setPointSize(24)
        self.onset_detection_label.setFont(font)
        self.onset_detection_label.setStyleSheet("border-radius : 10; border : 2px solid black; ")
        self.onset_detection_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.onset_detection_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.onset_detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.onset_detection_label.setObjectName("onset_detection_label")
        self.onset_openfile_btn = QtWidgets.QPushButton(self.tab_1)
        self.onset_openfile_btn.setEnabled(True)
        self.onset_openfile_btn.setGeometry(QtCore.QRect(450, 130, 100, 50))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/upload-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.onset_openfile_btn.setIcon(icon1)
        self.onset_openfile_btn.setCheckable(False)
        self.onset_openfile_btn.setDefault(False)
        self.onset_openfile_btn.setFlat(False)
        self.onset_openfile_btn.setObjectName("onset_openfile_btn")
        self.label = QtWidgets.QLabel(self.tab_1)
        self.label.setGeometry(QtCore.QRect(580, 10, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setUnderline(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.hop_size = QtWidgets.QLabel(self.tab_1)
        self.hop_size.setGeometry(QtCore.QRect(659, 86, 47, 13))
        self.hop_size.setObjectName("hop_size")
        self.normalizationCBK = QtWidgets.QCheckBox(self.tab_1)
        self.normalizationCBK.setEnabled(True)
        self.normalizationCBK.setGeometry(QtCore.QRect(640, 50, 91, 16))
        self.normalizationCBK.setObjectName("normalizationCBK")
        self.normalizationCBK.stateChanged.connect(self.checkBoxChangedAction)
        self.hop_size_input = QtWidgets.QLineEdit(self.tab_1)
        self.hop_size_input.setGeometry(QtCore.QRect(611, 85, 41, 20))
        self.hop_size_input.setObjectName("hop_size_input")
        self.get_onset_btn = QtWidgets.QPushButton(self.tab_1)
        self.get_onset_btn.setGeometry(QtCore.QRect(660, 130, 91, 41))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/two-rows-and-three-columns-layout.png"), QtGui.QIcon.Normal,
                        QtGui.QIcon.Off)
        self.get_onset_btn.setIcon(icon2)
        self.get_onset_btn.setObjectName("get_onset_btn")
        self.split_notes_btn = QtWidgets.QPushButton(self.tab_1)
        self.split_notes_btn.setEnabled(False)
        self.split_notes_btn.setGeometry(QtCore.QRect(660, 180, 91, 31))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon/tab-symbol.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.split_notes_btn.setIcon(icon3)
        self.split_notes_btn.setObjectName("split_notes_btn")
        self.onset_image_container = QtWidgets.QLabel(self.tab_1)
        self.onset_image_container.setGeometry(QtCore.QRect(50, 260, 701, 241))
        self.onset_image_container.setFrameShape(QtWidgets.QFrame.Box)
        self.onset_image_container.setText("")
        self.onset_image_container.setPixmap(QtGui.QPixmap("figure/placeholder.png"))
        self.onset_image_container.setScaledContents(True)
        self.onset_image_container.setAlignment(QtCore.Qt.AlignCenter)
        self.onset_image_container.setObjectName("onset_image_container")
        self.split_notes_pg = QtWidgets.QProgressBar(self.tab_1)
        self.split_notes_pg.setGeometry(QtCore.QRect(660, 220, 118, 23))
        self.split_notes_pg.setProperty("value", 0)
        self.split_notes_pg.setObjectName("split_notes_pg")
        self.predict_btn = QtWidgets.QPushButton(self.tab_1)
        self.predict_btn.setGeometry(QtCore.QRect(90, 520, 101, 41))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icon/096-idea.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.predict_btn.setIcon(icon5)
        self.predict_btn.setObjectName("predict_btn")
        self.predict_btn.setEnabled(True)
        self.play_btn = QtWidgets.QPushButton(self.tab_1)
        self.play_btn.setGeometry(QtCore.QRect(90, 620, 101, 41))
        self.play_btn.setEnabled(False)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icon/play-arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play_btn.setIcon(icon6)
        self.play_btn.setObjectName("play_btn")
        self.predict_notes_label = QtWidgets.QLabel(self.tab_1)
        self.predict_notes_label.setGeometry(QtCore.QRect(230, 520, 441, 141))
        self.predict_notes_label.setFrameShape(QtWidgets.QFrame.Box)
        self.predict_notes_label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.predict_notes_label.setTextFormat(QtCore.Qt.AutoText)
        self.predict_notes_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.predict_notes_label.setWordWrap(True)
        self.predict_notes_label.setIndent(-1)
        self.predict_notes_label.setObjectName("predict_notes_label")
        self.onset_filename_label = QtWidgets.QLineEdit(self.tab_1)
        self.onset_filename_label.setGeometry(QtCore.QRect(230, 130, 200, 50))
        self.onset_filename_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.onset_filename_label.setReadOnly(True)
        self.onset_filename_label.setObjectName("onset_filename_label")
        self.data_test_pg = QtWidgets.QProgressBar(self.tab_1)
        self.data_test_pg.setGeometry(QtCore.QRect(50, 530, 31, 23))
        self.data_test_pg.setProperty("value", 0)
        self.data_test_pg.setTextVisible(False)
        self.data_test_pg.setOrientation(QtCore.Qt.Vertical)
        self.data_test_pg.setObjectName("data_test_pg")
        self.notes_feature_extract = QtWidgets.QPushButton(self.tab_1)
        self.notes_feature_extract.setGeometry(QtCore.QRect(50, 210, 91, 41))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icon/068-zoom in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.notes_feature_extract.setIcon(icon7)
        self.notes_feature_extract.setObjectName("notes_feature_extract")
        self.notes_feature_extract.setEnabled(False)
        self.ket = QtWidgets.QLabel(self.tab_1)
        self.ket.setGeometry(QtCore.QRect(680, 520, 61, 16))
        self.ket.setObjectName("ket")
        self.ket_2 = QtWidgets.QLabel(self.tab_1)
        self.ket_2.setGeometry(QtCore.QRect(680, 540, 81, 16))
        self.ket_2.setObjectName("ket_2")
        self.ket_3 = QtWidgets.QLabel(self.tab_1)
        self.ket_3.setGeometry(QtCore.QRect(680, 560, 81, 16))
        self.ket_3.setObjectName("ket_3")
        self.ket_4 = QtWidgets.QLabel(self.tab_1)
        self.ket_4.setGeometry(QtCore.QRect(680, 580, 81, 16))
        self.ket_4.setObjectName("ket_4")
        self.ket_5 = QtWidgets.QLabel(self.tab_1)
        self.ket_5.setGeometry(QtCore.QRect(680, 600, 81, 16))
        self.ket_5.setObjectName("ket_5")
        self.ket_6 = QtWidgets.QLabel(self.tab_1)
        self.ket_6.setGeometry(QtCore.QRect(680, 620, 81, 16))
        self.ket_6.setObjectName("ket_6")
        self.ket_7 = QtWidgets.QLabel(self.tab_1)
        self.ket_7.setGeometry(QtCore.QRect(680, 640, 81, 16))
        self.ket_7.setObjectName("ket_7")
        self.sintesis_btn = QtWidgets.QPushButton(self.tab_1)
        self.sintesis_btn.setGeometry(QtCore.QRect(90, 570, 101, 41))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icon/022-speakers.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sintesis_btn.setIcon(icon8)
        self.sintesis_btn.setObjectName("sintesis_btn")
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.backpropagation_label = QtWidgets.QLabel(self.tab_4)
        self.backpropagation_label.setGeometry(QtCore.QRect(180, 10, 491, 41))
        font = QtGui.QFont()
        font.setFamily("Lato")
        font.setPointSize(24)
        self.backpropagation_label.setFont(font)
        self.backpropagation_label.setStyleSheet("border-radius : 10; border : 2px solid black; ")
        self.backpropagation_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.backpropagation_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.backpropagation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.backpropagation_label.setObjectName("backpropagation_label")
        self.max_epoch = QtWidgets.QLabel(self.tab_4)
        self.max_epoch.setGeometry(QtCore.QRect(526, 70, 71, 20))
        self.max_epoch.setAlignment(QtCore.Qt.AlignCenter)
        self.max_epoch.setObjectName("max_epoch")
        self.max_epoch_input = QtWidgets.QLineEdit(self.tab_4)
        self.max_epoch_input.setGeometry(QtCore.QRect(640, 70, 31, 20))
        self.max_epoch_input.setObjectName("max_epoch_input")
        self.learning_rate = QtWidgets.QLabel(self.tab_4)
        self.learning_rate.setGeometry(QtCore.QRect(530, 100, 71, 20))
        self.learning_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.learning_rate.setObjectName("learning_rate")
        self.hidden_layer = QtWidgets.QLabel(self.tab_4)
        self.hidden_layer.setGeometry(QtCore.QRect(528, 130, 71, 20))
        self.hidden_layer.setAlignment(QtCore.Qt.AlignCenter)
        self.hidden_layer.setObjectName("hidden_layer")
        self.learning_rate_input = QtWidgets.QLineEdit(self.tab_4)
        self.learning_rate_input.setGeometry(QtCore.QRect(640, 100, 31, 20))
        self.learning_rate_input.setObjectName("learning_rate_input")
        self.hidden_layer_input = QtWidgets.QLineEdit(self.tab_4)
        self.hidden_layer_input.setGeometry(QtCore.QRect(640, 130, 31, 20))
        self.hidden_layer_input.setObjectName("hidden_layer_input")
        self.train_btn = QtWidgets.QPushButton(self.tab_4)
        self.train_btn.setGeometry(QtCore.QRect(360, 202, 75, 31))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icon/066-line chart.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.train_btn.setIcon(icon7)
        self.train_btn.setObjectName("train_btn")
        self.save_model_btn = QtWidgets.QPushButton(self.tab_4)
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.setGeometry(QtCore.QRect(350, 610, 91, 31))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icon/download-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_model_btn.setIcon(icon8)
        self.save_model_btn.setObjectName("save_model_btn")
        self.train_pbar = QtWidgets.QProgressBar(self.tab_4)
        self.train_pbar.setGeometry(QtCore.QRect(250, 260, 321, 23))
        self.train_pbar.setProperty("value", 0)
        self.train_pbar.setObjectName("train_pbar")
        self.label_2 = QtWidgets.QLabel(self.tab_4)
        self.label_2.setGeometry(QtCore.QRect(567, 264, 47, 13))
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.data_frame_btn = QtWidgets.QPushButton(self.tab_3)
        self.data_frame_btn.setGeometry(QtCore.QRect(340, 120, 111, 51))
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icon/012-edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.data_frame_btn.setIcon(icon9)
        self.data_frame_btn.setCheckable(False)
        self.data_frame_btn.setChecked(False)
        self.data_frame_btn.setObjectName("data_frame_btn")
        self.df_pbar = QtWidgets.QProgressBar(self.tab_3)
        self.df_pbar.setGeometry(QtCore.QRect(280, 240, 241, 23))
        self.df_pbar.setProperty("value", 0)
        self.df_pbar.setOrientation(QtCore.Qt.Horizontal)
        self.df_pbar.setInvertedAppearance(False)
        self.df_pbar.setObjectName("df_pbar")
        self.data_saved_label = QtWidgets.QLabel(self.tab_3)
        self.data_saved_label.setGeometry(QtCore.QRect(270, 340, 241, 31))
        self.data_saved_label.setTextFormat(QtCore.Qt.AutoText)
        self.data_saved_label.setScaledContents(False)
        self.data_saved_label.setWordWrap(True)
        self.data_saved_label.setObjectName("data_saved_label")
        self.featureExtract_label = QtWidgets.QLabel(self.tab_3)
        self.featureExtract_label.setGeometry(QtCore.QRect(250, 10, 291, 41))
        font = QtGui.QFont()
        font.setFamily("Lato")
        font.setPointSize(24)
        self.featureExtract_label.setFont(font)
        self.featureExtract_label.setStyleSheet("border-radius : 10; border : 2px solid black; ")
        self.featureExtract_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.featureExtract_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.featureExtract_label.setAlignment(QtCore.Qt.AlignCenter)
        self.featureExtract_label.setObjectName("featureExtract_label")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(518, 244, 47, 13))
        self.label_3.setObjectName("label_3")
        self.show_data_frame = QtWidgets.QTableWidget(self.tab_3)
        self.show_data_frame.setGeometry(QtCore.QRect(35, 401, 731, 271))
        self.show_data_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.show_data_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.show_data_frame.setGridStyle(QtCore.Qt.DashLine)
        self.show_data_frame.setRowCount(0)
        self.show_data_frame.setColumnCount(0)
        self.show_data_frame.setObjectName("show_data_frame")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.onset_openfile_btn.clicked.connect(self.open_file_dialog)
        self.get_onset_btn.clicked.connect(self.get_onset)
        self.split_notes_btn.clicked.connect(self.split_notes)
        self.data_frame_btn.clicked.connect(self.make_data_frame)
        self.notes_feature_extract.clicked.connect(self.make_data_test)
        self.train_btn.clicked.connect(self.train_backpro)
        self.save_model_btn.clicked.connect(self.save_model)
        self.predict_btn.clicked.connect(self.predict_notes)
        self.sintesis_btn.clicked.connect(self.sintesis_audio)
        self.play_btn.clicked.connect(self.play_audio)

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
                self.play_btn.setEnabled(True)
            except:
                log.exception("error detected")

    def predict_notes(self):
        self.predict_notes_label.clear()
        global predicted_notes, class_predict
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
            else:
                predicted_notes.append('t')
        print(answer)
        class_predict = answer
        for note in predicted_notes:
            notes += note + ' . '
        self.predict_notes_label.setText(notes)
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
        self.onset_filename_label.setText(filename)

    def openFileNamesDialog(self):
        global filename
        path = QtWidgets.QFileDialog.getOpenFileName(directory='dataset/data_val/')
        filename = path[0]

    def get_onset(self):
        global filename, normalization, onset_times_backtrack, onset_sample
        hop_size = self.hop_size_input.text()
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
                onset_times_backtrack, onset_sample = onsetDetection(x, sr, hop_size)
            else:
                onset_times_backtrack, onset_sample = onsetDetectionNonNormalize(x, sr, hop_size)
            plotingWave(x, sr, filename, hop_size, onset_times_backtrack, normalize=normalization)
            self.onset_image_container.setPixmap(QtGui.QPixmap("figure/onset.png"))
            self.split_notes_btn.setEnabled(True)
            self.data_test_pg.setProperty("value", 0)
        print("onset_sample {}, panjang {}".format(onset_sample, len(onset_sample)))

    def split_notes(self):
        self.calc = MakeWav()
        self.calc.countChanged.connect(self.countSplitNotes)
        self.calc.start()
        self.notes_feature_extract.setEnabled(True)

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
        n_epoch = self.max_epoch_input.text()
        n_lrate = self.learning_rate_input.text()
        n_hidden = self.hidden_layer_input.text()

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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Kendang App"))
        self.onset_detection_label.setText(_translate("MainWindow", "Onset Detection"))
        self.onset_openfile_btn.setText(_translate("MainWindow", "open file"))
        self.label.setText(_translate("MainWindow", "onset detection parameter"))
        self.hop_size.setText(_translate("MainWindow", "Hop Size"))
        self.normalizationCBK.setText(_translate("MainWindow", "Normalization"))
        self.get_onset_btn.setText(_translate("MainWindow", " get onset"))
        self.split_notes_btn.setText(_translate("MainWindow", " segmentation"))
        self.predict_btn.setText(_translate("MainWindow", "predict notes"))
        self.play_btn.setText(_translate("MainWindow", "play"))
        self.predict_notes_label.setText(_translate("MainWindow", "noted appears here"))
        self.onset_filename_label.setText(_translate("MainWindow", "<filename>"))
        self.notes_feature_extract.setText(_translate("MainWindow", "get feature"))
        self.ket.setText(_translate("MainWindow", "Ket : "))
        self.ket_2.setText(_translate("MainWindow", "C - Cung Kanan"))
        self.ket_3.setText(_translate("MainWindow", "D - De Kanan"))
        self.ket_4.setText(_translate("MainWindow", "T - Tek Kanan"))
        self.ket_5.setText(_translate("MainWindow", "p - Pak Kiri"))
        self.ket_6.setText(_translate("MainWindow", "u - Pung Kiri"))
        self.ket_7.setText(_translate("MainWindow", "t - Teng Kiri"))
        self.sintesis_btn.setText(_translate("MainWindow", "synthesis"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "onset_detection"))
        self.backpropagation_label.setText(_translate("MainWindow", "Backpropagation Neural Network"))
        self.max_epoch.setText(_translate("MainWindow", "max epoch"))
        self.learning_rate.setText(_translate("MainWindow", "learning rate"))
        self.hidden_layer.setText(_translate("MainWindow", "hidden layer"))
        self.train_btn.setText(_translate("MainWindow", "train"))
        self.save_model_btn.setText(_translate("MainWindow", "save model"))
        self.label_2.setText(_translate("MainWindow", "progress"))
        self.img_backpro = QtWidgets.QLabel(self.tab_4)
        self.img_backpro.setGeometry(QtCore.QRect(160, 310, 481, 271))
        self.img_backpro.setText("")
        self.img_backpro.setPixmap(QtGui.QPixmap("asset/img_backpro.jpeg"))
        self.img_backpro.setScaledContents(True)
        self.img_backpro.setObjectName("img_backpro")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Backpropogation"))
        self.data_frame_btn.setText(_translate("MainWindow", "make data frame"))
        self.data_saved_label.setText(_translate("MainWindow", "data train saved at :"))
        self.featureExtract_label.setText(_translate("MainWindow", "Feature Extraction"))
        self.label_3.setText(_translate("MainWindow", "progress"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Feature Extraction"))

    def countSplitNotes(self, value):
        self.split_notes_pg.setValue(value)

    def countTrain(self, value):
        self.train_pbar.setValue(value)

    def countDataFrame(self, value):
        self.df_pbar.setValue(value)

    def countDataTest(self, value):
        self.data_test_pg.setValue(value)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    file = open('theme/Adaptic.qss', 'r')
    with file:
        qss = file.read()
        app.setStyleSheet(qss)
    # apply_stylesheet(app, theme='dark_amber.xml')

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
