import deliverable_model as dm

model = dm.load(
    "./res/deliverable_model"
)

request = dm.make_request(query=["帮我看看上海26号会下雨吗", "青岛今天天气怎么样", "你以后就叫奥特曼", "打开NFC界面。", "附近哪里有KFC"])

result = model.inference(request)
for i in result.data:
    print(i)
    print('\n')


