import re
import telegram


def cut_command_text(command_text: str):
    t = command_text.strip()
    reg1 = r"^/[\w_]+@[\w_]+bot\s"
    reg2 = r"^/[\w_]+\s"
    m1 = re.match(reg1, t)
    m2 = re.match(reg2, t)
    if m1:
        return t[m1.span()[1] :]
    else:
        return t[m2.span()[1] :]


def replace_ai2shinnku(content: str):
    content = content.replace("一个AI助手", "ai真红")
    content = content.replace("AI助手", "ai真红")
    content = content.replace("人工智能助手", "ai真红")
    content = content.replace("AI语言模型", "ai真红")
    content = content.replace("AI 聊天机器人", "ai真红")
    return content


async def get_reply_list(bot: telegram.Bot, update: telegram.Update):
    message_id = update.message.message_id
    text = update.message.text
    pass
