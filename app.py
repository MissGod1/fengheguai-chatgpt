#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Time    :   2023/04/20 10:23:18
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import warnings
warnings.filterwarnings('ignore')
import sys
import os
import io
import logging
import threading
import gc

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal
from ui.Ui_app import Ui_Form

import pyaudio
import wave
import sounddevice as sd

from vits_uma_genshin_honkai.export import VITS
from chatgpt import ChatGPT
from baidu import NewTTSExecutor, NewASRExecutor
import paddle

from dotenv.main import load_dotenv

_translate = QtCore.QCoreApplication.translate
# load environment variable
load_dotenv()

# set log level
logger = logging.getLogger('PaddleSpeech')
logger.setLevel(getattr(logging, os.environ['LOG_LEVEL']))

# chatgpt setting
API_TOKEN = os.environ['CHATGPT_API_TOKEN']
proxies = {
    'http': os.environ['HTTP_PROXY'],
    'https': os.environ['HTTPS_PROXY'],
}

# vits setting
vits_config = os.environ['VITS_CONFIG']
vits_model_path = os.environ['VITS_MODEL_PATH']

decoder_model = ['VITS', 'PaddleSpeech']


class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    load_signal = pyqtSignal()
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.audio = None
        self.stream = None
        self.data = None
        self.device = 'cuda:0'
        
        self.asr = NewASRExecutor()
        self.tts = VITS(config=vits_config, 
                         model_path=vits_model_path)
        self.chat = ChatGPT(API_TOKEN)
        self.speaker_id = 0
        
        
        self.load_signal.connect(self.load_model_finish)
        
        self.btn_listen.setEnabled(False)
        self.btn_listen.pressed.connect(self.listen_start)
        self.btn_listen.released.connect(self.listen_stop)
        
        self.cb_speaker.addItems(self.tts.speakers)
        self.cb_speaker.currentIndexChanged.connect(self.change_speaker)
        
        self.cb_decoder.addItems(decoder_model)
        self.cb_decoder.currentIndexChanged.connect(self.change_decoder)
        
        self.btn_load.clicked.connect(self.load_model)
        
        self.lb_status.setText(_translate('Form', 'please load model'))
    
    def load_model(self):
        self.lb_status.setText(_translate('Form', 'loading ...'))
        def load():
            self.asr._init_from_path('conformer_u2pp_online_wenetspeech')
            self.tts._init_from_path()
            self.load_signal.emit()

        th = threading.Thread(target=load)
        th.setDaemon(True)
        th.start()
    
    def load_model_finish(self):
        self.lb_status.setText(_translate('Form', 'load finish.'))
        self.btn_listen.setEnabled(True)
        
    def change_decoder(self, i):
        self.btn_listen.setEnabled(False)
        self.lb_status.setText(_translate('Form', 'please load model'))
        if i == 1:
            self.tts = NewTTSExecutor()
            self.device = 'gpu:0'
        else:
            self.tts = VITS(config=vits_config, model_path=vits_model_path,
                            sid=self.cb_speaker.currentIndex())
            self.device = 'cuda:0'
        gc.collect()
        paddle.device.cuda.empty_cache()
    
    def change_speaker(self, i):
        if hasattr(self.tts, 'sid'):
            self.tts.sid = i
    
    def listen_start(self):
        logger.debug('Start Listen....')
        self.btn_listen.setStyleSheet("background-color: rgb(0, 255, 127);")
        self.data = []
        def callback(in_data, frame_count, time_info, status):
            self.data.append(in_data)
            return b'', pyaudio.paContinue
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    stream_callback=callback
                )
        self.stream.start_stream()
        pass
    
    def listen_stop(self):
        logger.debug('Stop Listen...')
        self.btn_listen.setStyleSheet("")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        wav_data = b''.join(self.data)
        output = io.BytesIO()
        # output.seek(0)
        with wave.open(output, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(wav_data)
        
        th = threading.Thread(target=self.process_handler, args=(output,))
        th.setDaemon(True)
        th.start()
        pass
    
    def process_handler(self, output):
        logger.info("Speech to Text...")
        res = self.asr(audio_file=output, device="gpu:0")
        logger.info(f'STT: {res}')
        
        if res != '':
            # logger.info("ChatGPT...")
            # res = self.chat(res)
            # logger.info(f'ChatGPT: {res}')
            
            logger.info("Text to Speech...")
            audio, fs = self.tts(text=res, device=self.device)
            sd.play(audio, fs)
            
        gc.collect()
        paddle.device.cuda.empty_cache()

if __name__ == '__main__':
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    