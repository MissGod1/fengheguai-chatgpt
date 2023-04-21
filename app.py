# import gevent
# from gevent import monkey
# monkey.patch_all()
from PyQt5 import QtCore, QtGui, QtWidgets
from Ui_app import Ui_Form
import sys
import pyaudio
import wave
from paddlespeech.cli.asr.infer import ASRExecutor
# from paddlespeech.cli.tts.infer import TTSExecutor
from vits_uma_genshin_honkai.run import VITS
from winsound import PlaySound
import requests
import json
import logging
import threading
from dotenv.main import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

logger = logging.getLogger('PaddleSpeech')
logger.setLevel(getattr(logging, os.environ['LOG_LEVEL']))

API_TOKEN=os.environ['CHATGPT_API_TOKEN']
proxies = {
    'http': os.environ['HTTP_PROXY'],
    'https': os.environ['HTTPS_PROXY'],
}


class ChatGPT(object):
    def __init__(self, api_token=None) -> None:
        self.api_token=api_token
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_token
        }
        
    def request(self, content):
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": content}]
        }
        
        try:
            req = requests.post('https://api.openai.com/v1/chat/completions', 
                            headers=self.headers,
                            json=data, proxies=proxies)
            response = req.text
            logger.debug(response)
            response = json.loads(response)
            return response['choices'][0]['message']['content']
        except Exception as e:
            return e
    
    def __call__(self, content):
        return self.request(content)


class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.audio = None
        self.stream = None
        self.data = None
        self.output = 'tmp.wav'
        
        self.asr = ASRExecutor()
        # self.tts = TTSExecutor()
        self.vits = VITS(config='vits_uma_genshin_honkai/model/config.json', 
                         model_path='vits_uma_genshin_honkai/model/G_953000.pth')
        self.chat = ChatGPT(API_TOKEN)
        self.speaker_id = 0
        
        self.BtnListen.pressed.connect(self.listen_start)
        self.BtnListen.released.connect(self.listen_stop)
        
        self.CBSpeaker.addItems(self.vits.speakers)
        self.CBSpeaker.currentIndexChanged.connect(self.change_speaker)
        self.init_model()
    
    def init_model(self):
        self.asr._init_from_path('conformer_u2pp_online_wenetspeech')
        # self.tts._init_from_path()
        # wakeup_file = 'zh.wav'
        # logger.debug("Speech to Text...")
        # res = self.asr(audio_file=wakeup_file, device="gpu:0")
        # logger.debug('STT: {res}')
        # logger.debug("Text to Speech...")
        # self.tts(text=res, output=wakeup_file, device="gpu:0")
        # PlaySound(wakeup_file, flags=0)
    
    def change_speaker(self, i):
        self.speaker_id = i
    
    def listen_start(self):
        logger.debug('Start Listen....')
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
        wav_data = b''.join(self.data)
        with wave.open(self.output, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(wav_data)
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        th = threading.Thread(target=self.process_handler)
        th.setDaemon(True)
        th.start()
        # self.process_handler()
        # gevent.spawn(self.process_handler)
        pass
    
    def process_handler(self):
        logger.info("Speech to Text...")
        res = self.asr(audio_file=self.output, device="gpu:0")
        logger.info(f'STT: {res}')
        if res != '':
            logger.info("ChatGPT...")
            res = self.chat(res)
            logger.info(f'ChatGPT: {res}')
            logger.info("Text to Speech...")
            # self.tts(text=res, output=self.output, device="gpu:0")
            self.vits(res, self.output, self.speaker_id)
            PlaySound(self.output, flags=0)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    