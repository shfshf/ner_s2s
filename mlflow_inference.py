import requests
import pandas as pd

# 构造需要进行推断的数据
cols = ["a", "b", "c"]
data = [[1, 2, 3]]
model_input = pd.DataFrame(data, columns=cols)

# 指定ip, 端口
url = "http://127.0.0.1:5000/invocations"

# 传递的参数需要从dataframe转化为json格式
req_data = model_input.to_json(orient='split')

# 指定headers参数
headers = {'content-type': 'application/json; format=pandas-split'}

# 使用POST方式调用REST api
respond = requests.request("POST", url, data=req_data, headers=headers)

# 获取返回值
respond.json()
