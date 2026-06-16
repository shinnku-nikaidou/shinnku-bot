# encoding: utf-8
import os
from logging import getLogger

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from constants.craw import user_agent
from utils.decorators import send_action
from utils.text_handling import cut_command_text

# Init logger

logger = getLogger(__name__)


async def post_netease(query: str) -> dict:
    url = "https://netease.project.ac.cn/search"
    params = {"keywords": query, "limit": 1}
    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    logger.debug("search result: %s", response.content)
    return response.json()


async def get_download_link(song_id: int) -> str:
    url = "https://netease.project.ac.cn/song/url"
    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params={"id": song_id}, headers=headers)
    response.raise_for_status()
    logger.debug("link result: %s", response.content)
    return response.json()["data"][0]["url"]


async def download_music(url: str, music_path: str) -> None:
    headers = {"User-Agent": user_agent, "Referer": "http://music.163.com/"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url=url, headers=headers)
    response.raise_for_status()
    os.makedirs(os.path.dirname(music_path), exist_ok=True)
    with open(music_path, "wb") as f:
        f.write(response.content)
    logger.info("%s 下载成功", music_path)


@send_action(ChatAction.TYPING)
async def netease(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    query = cut_command_text(update.message.text)
    data = await post_netease(query)
    song_info = data["result"]["songs"][0]
    song_id = int(song_info["id"])
    name = song_info["name"]
    logger.debug("song id: %s", song_id)
    url = await get_download_link(song_id)
    ext = "mp3" if url.endswith("mp3") else "flac"
    music_path = f"./data/netease/{name}.{ext}"
    await download_music(url, music_path)
    with open(music_path, "rb") as f:
        await update.message.reply_audio(audio=f)
    return music_path
