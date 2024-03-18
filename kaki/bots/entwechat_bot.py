import requests
import json


# 发送文本消息
def send_text(webhook, content, mentioned_list=None, mentioned_mobile_list=None):
    header = {
                "Content-Type": "application/json",
                "Charset": "UTF-8"
                }
    data ={

        "msgtype": "text",
        "text": {
            "content": content
            ,"mentioned_list":mentioned_list
            ,"mentioned_mobile_list":mentioned_mobile_list
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)


# 发送markdown消息
def send_md(webhook, content):
    header = {
                "Content-Type": "application/json",
                "Charset": "UTF-8"
                }
    data ={

        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)


if __name__ == "__main__":
    webhook = "****"
    send_text(webhook, content='HsuHeinrich', mentioned_mobile_list=[1])
    send_md(webhook, content='# 一级标题 \n 微信搜索HsuHeinrich,发现更多精彩 ')