# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text
from constants.craw import user_agent
import os
import requests

# Init logger

logger = getLogger(__name__)


def post_netease(query: str) -> dict:
    url = "https://netease.project.ac.cn/search"
    params = {"keywords": query, "limit": 1}
    headers = {"User-Agent": user_agent}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    logger.debug("search result: %s", response.content)
    return response.json()


def get_download_link(song_id: int) -> str:
    url = "https://netease.project.ac.cn/song/url"
    headers = {"User-Agent": user_agent}
    response = requests.get(url, params={"id": song_id}, headers=headers, timeout=10)
    response.raise_for_status()
    logger.debug("link result: %s", response.content)
    return response.json()["data"][0]["url"]


def download_music(url: str, music_path: str) -> None:
    headers = {"User-Agent": user_agent, "Referer": "http://music.163.com/"}
    response = requests.get(url=url, headers=headers, timeout=60)
    response.raise_for_status()
    os.makedirs(os.path.dirname(music_path), exist_ok=True)
    with open(music_path, "wb") as f:
        f.write(response.content)
    logger.info("%s 下载成功", music_path)


@send_action(ChatAction.TYPING)
async def netease(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    query = cut_command_text(update.message.text)
    data = post_netease(query)
    song_info = data["result"]["songs"][0]
    song_id = int(song_info["id"])
    name = song_info["name"]
    logger.debug("song id: %s", song_id)
    url = get_download_link(song_id)
    ext = "mp3" if url.endswith("mp3") else "flac"
    music_path = f"./data/netease/{name}.{ext}"
    download_music(url, music_path)
    with open(music_path, "rb") as f:
        await update.message.reply_audio(audio=f)
