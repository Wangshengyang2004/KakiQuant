import requests
import json
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider, internet 
from random import randint

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



# 构造pandas数据
# 自定义fake
fake = Faker('zh_CN')
class MyProvider(BaseProvider):
    def myCityLevel(self):
        cl = ["一线", "二线", "三线", "四线+"]
        return cl[randint(0, len(cl) - 1)]
    def myGender(self):
        g = ['F', 'M']
        return g[randint(0, len(g) - 1)]
    def myDevice(self):
        d = ['Ios', 'Android']
        return d[randint(0, len(d) - 1)]
fake.add_provider(MyProvider)

# 构造假数据
uid=[]
cityLevel=[]
gender=[]
device=[]
age=[]
activeDays=[]
for i in range(10):
    uid.append(i+1)
    cityLevel.append(fake.myCityLevel())
    gender.append(fake.myGender())
    device.append(fake.myDevice())
    age.append(fake.random_int(min=18, max=65))
    activeDays.append(fake.random_int(min=0, max=180))

raw_data= pd.DataFrame({'uid':uid,
                        'cityLevel':cityLevel,
                        'gender':gender,
                        'device':device,
                        'age':age,
                        'activeDays':activeDays,
                       })

# 通过style美化df
# 增加色阶、标题，隐藏索引、uid列
title = '活跃统计'
df = raw_data.style\
    .background_gradient(cmap='Pastel1',subset=['activeDays'])\
    .set_caption(title)\
    .hide_index()\
    .hide_columns(subset=['uid'])\
    .to_excel('file_demo.xlsx', engine='openpyxl', index=False)

# 发送文件
def send_file(webhook, file):
    # 获取media_id
    key = webhook.split('key=')[1]
    id_url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file'
    files = {'file': open(file, 'rb')}
    res = requests.post(url=id_url, files=files)
    media_id = res.json()['media_id']

    header = {
                "Content-Type": "application/json",
                "Charset": "UTF-8"
                }
    data ={
    "msgtype": "file",
    "file": {
                "media_id": media_id
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)



if __name__ == "__main__":
    webhook = "****"
    send_text(webhook, content='HsuHeinrich', mentioned_mobile_list=[1])
    send_md(webhook, content='# 一级标题 \n 微信搜索HsuHeinrich,发现更多精彩 ')