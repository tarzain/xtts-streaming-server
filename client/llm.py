import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Iterator

import requests
import asyncio
import websockets
import openai

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

# Main execution
if __name__ == "__main__":
    with no_ssl_verification():
        openai.base_url = "http://localhost:8000/v1/"
        # Example of an OpenAI ChatCompletion request with stream=True
        # https://platform.openai.com/docs/guides/chat

        # a ChatCompletion request
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': "What's 1+1? Answer in one word."}
            ],
            temperature=0,
            stream=True  # this time, we set stream=True
        )

        for chunk in response:
            print(chunk)