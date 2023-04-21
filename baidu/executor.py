#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   executor.py
@Time    :   2023/04/21 18:10:34
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import sys
import paddle
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor

class NewASRExecutor(ASRExecutor):
    def __call__(self,
                 audio_file,
                 model: str='conformer_u2pp_online_wenetspeech',
                 sample_rate: int=16000,
                 force_yes: bool=False,
                 device=paddle.get_device()):
        """
        Python API to call an executor.
        """
        paddle.set_device(device)

        if not self._check(audio_file, sample_rate, force_yes):
            sys.exit(-1)

        self.preprocess(model, audio_file)
        self.infer(model)
        res = self.postprocess()  # Retrieve result of asr.

        return res


class NewTTSExecutor(TTSExecutor):
    def postprocess(self, output):
        return (self._outputs['wav'].numpy(), self.am_config.fs)