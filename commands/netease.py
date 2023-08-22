# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from constants.craw import user_agent
import json
import re
import time
import os
import urllib.request
import subprocess
import requests

# Init logger

logger = getLogger(__name__)


def postnetease(scr):
    target = "https://netease.project.ac.cn/search?keywords="
    plus = "&limit=1"
    header = {
        'User-Agent': user_agent,
    }
    respond = requests.get(target+scr+plus, headers=header)
    print(respond.content)
    return json.loads(respond.content)

def get_download_link(_id:int):
    target = "https://netease.project.ac.cn/song/url?id="
    header = {
        'User-Agent': user_agent,
    }
    respond = requests.get(target+str(_id), headers=header)
    print(respond.content)
    return json.loads(respond.content)["data"][0]["url"]

def download_music(url, music_path):
    headers={
        'User-Agent':user_agent,
        'Referer':'http://music.163.com/'
    }
    response = requests.get(url=url, headers=headers)
    music_data = response.content
    with open(music_path, 'wb') as f:
        f.write(music_data)
        print(music_path, '下载成功')

@send_action(ChatAction.TYPING)
async def netease(update: Update,context: ContextTypes.DEFAULT_TYPE) -> str:
    t = str(update.message.text[8:]).strip()
    a = postnetease(t)
    print(a)
    _id = int(a['result']['songs'][0]['id'])
    name = a['result']['songs'][0]['name']
    print(_id)
    url = get_download_link(_id)
    mp3_path = f"./data/netease/{name}.mp3"
    download_music(url, mp3_path)
    # subprocess.call(["ffmpeg", "-n", "-i", mp3_path, "-acodec", "aac", "-ac", "2", "-ar", "44100", aac_path])
    await update.message.reply_audio(audio=open(mp3_path, 'rb'))
