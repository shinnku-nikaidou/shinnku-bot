import re

def cut_command_text(command_text: str):
    t = command_text.strip()
    reg1 = r'^/[\w_]+@[\w_]+bot\s'
    reg2 = r'^/[\w_]+\s'
    m1 = re.match(reg1, t)
    m2 = re.match(reg2, t)
    if m1:
        return t[m1.span()[1]:]
    else:
        return t[m2.span()[1]:]
