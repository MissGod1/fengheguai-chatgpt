# coding=utf-8
import time
import os
# import gradio as gr
from . import utils, commons
import argparse
# import .commons
from .models import SynthesizerTrn
from .text import text_to_sequence
import torch
from torch import no_grad, LongTensor
from scipy.io import wavfile
import numpy as np
# import webbrowser
import logging
# import gradio.processing_utils as gr_processing_utils
logging.getLogger('numba').setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces

# audio_postprocess_ori = gr.Audio.postprocess
# def audio_postprocess(self, y):
#     data = audio_postprocess_ori(self, y)
#     if data is None:
#         return None
#     return gr_processing_utils.encode_url_or_file_to_base64(data["name"])
# gr.Audio.postprocess = audio_postprocess

def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols,  hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

def vits(text, language, speaker_id, noise_scale=0.6, noise_scale_w=0.668, length_scale=1.2):
    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 100 and limitation:
        return f"输入文字过长！{len(text)}>100", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = LongTensor([speaker_id]).to(device)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"

def search_speaker(search_value):
    for s in speakers:
        if search_value == s:
            return s
    for s in speakers:
        if search_value in s:
            return s

def change_lang(language):
    if language == 0:
        return 0.6, 0.668, 1.2
    else:
        return 0.6, 0.668, 1.1


class VITS(object):
    def __init__(self, device='cuda:0', 
                 config='./model/config.json',
                 model_path='./model/G_953000.pth',
                 sid=0,
                 lang=0) -> None:
        self.device = device
        self.sid = sid
        self.model_path = model_path
        self.lang = lang
        self.hps_ms = utils.get_hparams_from_file(config)
        self.speakers: list = self.hps_ms.speakers
        
    def vits(self, text, language, speaker_id, noise_scale=0.6, noise_scale_w=0.668, length_scale=1.4):
        start = time.perf_counter()
        if not len(text):
            return "输入文本不能为空！", None, None
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
        if len(text) > 100 and limitation:
            return f"输入文字过长！{len(text)}>100", None, None
        if language == 0:
            text = f"[ZH]{text}[ZH]"
        elif language == 1:
            text = f"[JA]{text}[JA]"
        else:
            text = f"{text}"
        stn_tst, clean_text = get_text(text, self.hps_ms)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            speaker_id = LongTensor([speaker_id]).to(self.device)
            audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale,        
                                        noise_scale_w=noise_scale_w,
                                        length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"
        
    def __call__(self, text, device):
        status, (fs, data), msg = self.vits(text, self.lang, self.sid)
        logging.getLogger('PaddleSpeech').debug(f'{status}: {msg}')

        return data, fs
    
    def _init_from_path(self):
        self.net_g_ms = SynthesizerTrn(
                            len(self.hps_ms.symbols),
                            self.hps_ms.data.filter_length // 2 + 1,
                            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                            n_speakers=self.hps_ms.data.n_speakers,
                            **self.hps_ms.model)
        _ = self.net_g_ms.eval().to(self.device)
        _, _, _, _ = utils.load_checkpoint(self.model_path, self.net_g_ms, None)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--text', type=str, default='你怎么回事啊大兄弟')
    parser.add_argument('--cleaner',type=str, default='')
    parser.add_argument('--lang', type=int, default=0)
    parser.add_argument('--speaker', type=str, default='凯亚')
    args = parser.parse_args()
    device = torch.device(args.device)
    
    hps_ms = utils.get_hparams_from_file(r'./model/config.json')
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval().to(device)
    speakers: list = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(r'./model/G_953000.pth', net_g_ms, None)

    sid = speakers.index(args.speaker)
    
    status, (_, data), msg = vits(args.text, args.lang, sid)
    print(status, data, msg)
    from scipy.io import wavfile
    import numpy as np
    data *= 32767 / max(1e-8, np.max(np.abs(data))) * 0.6
    # data = data / max(1e-8, np.abs(data).max())
    # data = data * 32767
    data = data.astype(np.int16)
    
    wavfile.write('output.wav', hps_ms.data.sampling_rate, data)
    
    from winsound import PlaySound
    PlaySound('output.wav', flags=0)
