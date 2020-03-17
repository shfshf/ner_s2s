import deliverable_model as dm

model = dm.load(
    "/Users/shf/PycharmProjects/ner_s2s/results/deliverable_model"
)

request = dm.make_request(query=[[i for i in "帮我看看上海26号会下雨吗"], [i for i in "以后叫你小倩。"],
                                 [i for i in "以后你就叫二傻吧"], [i for i in "明天北京天气怎么样？"]])
# request = dm.make_request(query=["查询明天的天气", "明天黑龙江天气如何"])
result = model.inference(request)
for i in result.data:
    print(i)
    print('\n')


