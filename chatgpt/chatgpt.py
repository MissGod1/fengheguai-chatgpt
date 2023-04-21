#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   chatgpt.py
@Time    :   2023/04/21 18:04:31
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import requests
import json
import logging

logger = logging.getLogger('PaddlePaddle')

class ChatGPT(object):
    def __init__(self, api_token=None) -> None:
        self.api_token=api_token
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_token
        }
        
    def request(self, content, proxies=None):
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
    
    def __call__(self, content, proxies=None):
        return self.request(content, proxies)