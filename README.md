# 模型训练及推理

ner_s2s 是基于tensorflow的NER（命名实体识别）训练、推理模型。代码github链接[*ner_s2s*](https://github.com/shfshf/ner_s2s)

### ner_s2s项目文件 
* data文件夹用来存放数据集，其中包括训练集（train.conllx）、测试集（test.conllx）、实体（entity.txt）、字典（unicode_char_list.txt）等。
* docker文件夹用来制作项目的docker镜像，以便于后面的训练部署，其中nightly是测试版，stable是稳定版，都包含制作了两个镜像：trainer是训练镜像，server是服务镜像。
* tests文件夹是pytest测试脚本，对项目的部分脚本进行测试验证。
* ner_s2s文件夹是项目具体的执行代码块，程序的执行入口有两个：
基于tensorflow的estimator框架--ner_estimator文件夹下的estimator_run文件;
基于tensorflow的keras框架--ner_keras文件夹下的keras_run文件。二者都可以执行项目文件的训练任务。
* configure.yaml 是项目文件的超参数配置。
* Makefile 是项目文件部分功能执行的命令集，比如docker打包生成的镜像文件命令等。

### configure.yaml 
其中:  
* Data source 部分指定了项目文件的数据（data）的相对路径：包括训练集、测试集、实体、字典等，其中warm_start_dir指定了checkpoint需要继续训练的文件路径；
* model configure 部分指定了一些常用模型的超参数配置（包括estimator与keras模式），例如batch_size，epochs，learning_rate等
* Data output 部分指定了模型生成的多种保存格式的相对路径，包括pb，h5等，其中deliverable_model对模型进行了前置，后置处理，推理、解码进行了封装

## 安装 ner_s2s 项目环境
* 在conda环境中新建python=3.6版本，名为ner_s2s的conda环境:
bash
```
conda create --name ner_s2s python=3.6
``` 
* 安装 ner_s2s 项目环境依赖:
bash
```
pip install -U ner_s2s
``` 
* 安装 tensorflow-gpu 环境版本（如有nvidia gpu芯片）

这里需要说明的是，通过`pip install -U ner_s2s`安装所需的依赖在服务器上的conda环境中，
是无法直接使用gpu训练模型的，需要在conda环境中安装 nvidia 相关的 cudnn 驱动；

所以需要在 conda 环境中多安装一次 gpu 驱动：
bash
```
# 使用中科大conda加速镜像
conda config –add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free
conda config –set show_channel_urls yes
# 通过conda安装tensorflow-gpu版本
conda install tensorflow-gpu==1.15
```
怎样查看 conda 环境是否可以使用gpu加速？三步：
bash
* `python` 进入命令行模式
* `import tensorfow as tf` 导入tensorflow
* `tf.test.is_gpu_available()` 如果显示 True 则大功告成

## ner_s2s 训练及推理命令

### ner_s2s 训练(通过项目代码) 
### cpu (没有GPU加速)
bash
```
# estimator 模式
python -m ner_s2s.ner_estimator.estimator_run
# keras 模式
python -m ner_s2s.ner_keras.keras_run
```
### gpu 指定显卡运行(针对多显卡)
bash
```
# estimator 模式
CUDA_VISIBLE_DEVICES=0 python -m ner_s2s.ner_estimator.estimator_run
# keras 模式
CUDA_VISIBLE_DEVICES=1 python -m ner_s2s.ner_keras.keras_run
```
### ner_s2s 推理(通过项目代码)
推理项目代码执行入口在 ner_s2s/server/http.py 文件
### run service
bash
```
python -m ner_s2s.server.http /path/to/saved_model
```
其中 /path/to/saved_model 是ner_s2s训练的结果results/deliverable_model的绝对路径。例如：
bash
```
python -m ner_s2s.server.http /Users/shf/PycharmProjects/ner_s2s/results/deliverable_model 
```
默认启动在 主机： `localhost` 端口：`5000`
### input format（在浏览器的ip地址输入）
example:
```
http://localhost:5000/parse?q=明天上海的天气怎么样？
```
### output
example:
```json
{
    "ents": [
        "城市名",
        "日期"
    ],
    "spans": [
        {
            "end": 2,
            "start": 0,
            "type": "日期"
        },
        {
            "end": 4,
            "start": 2,
            "type": "城市名"
        }
    ],
    "text": "明天上海的天气怎么样？"
}
```
## 制作docker镜像 

通过docker镜像对ner_s2s进行训练与推理
### Makefile
docker打包镜像的入口命令在根目录文件夹下的 Makefile 文件中，其中用到的make命令如下：
```
.PHONY: build_docker_nightly_build
build_docker_nightly_build: build_docker_nightly_build_trainer build_docker_nightly_build_server

.PHONY: build_docker_nightly_build_trainer
build_docker_nightly_build_trainer: dist
	cp -r dist docker/nightly/trainer/
	docker rmi -f ner_trainer
	docker build --no-cache --force-rm --tag ner_trainer --file docker/nightly/trainer/Dockerfile docker/nightly/trainer/

.PHONY: build_docker_nightly_build_server
build_docker_nightly_build_server: dist
	cp -r dist docker/nightly/server/
	docker rmi -f ner_server
	docker build --no-cache --force-rm --tag ner_server --file docker/nightly/server/Dockerfile docker/nightly/server/
```
首先要开启docker服务，然后在terminal中输入命令（在根目录下）：
bash
```
make build_docker_nightly_build
```
这就是上面 Makefile 文件的第一行命令，即可执行下面的两个make命令（build_docker_nightly_build_trainer与build_docker_nightly_build_server），
分别生成两个docker镜像文件：ner_trainer 与 ner_server；

这里如果需要修改docker镜像的文件名，只需要在上面的两个make命令中分别修改即可,比如将ner_trainer修改为 shf_ner_trainer:
```
.PHONY: build_docker_nightly_build_trainer
build_docker_nightly_build_trainer: dist
	cp -r dist docker/nightly/trainer/
	docker rmi -f shf_ner_trainer
	docker build --no-cache --force-rm --tag shf_ner_trainer --file docker/nightly/trainer/Dockerfile docker/nightly/trainer/
```
其他make命令无需改变。

执行结果：
```
shf@tesla:~$ docker images
REPOSITORY         TAG       IMAGE ID         CREATED          SIZE
ner_server        latest    be9fa79c6565     9 seconds ago     4.4GB
shf_ner_trainer   latest    545cd33fa016     2 minutes ago     4.4GB
```


## 小样本增量学习
* 通过训练一个比较全面数据（all_domain）的准确的大模型为基础，即利用大模型生成的checkpoint作为新模型的读入值
* 增加一个全新的domain，数据是少量的全新数据，增加的新实体值直接添加到entity.txt的后面即可
* 将前面（all_domain）的每个domain随机挑出与新domain等量的数据，合并到一起为新domain的数据集
* 新domain的数据集包含合并后的训练集(train.conllx)、测试集(test.conllx)、增加新实体后的(entity.txt)、字典（unicode_char_list.txt）等
### 给神经网络层命名
(ner_s2s.ner_estimator.algorithms.model.py)中增加两个函数来命名网络层的结构
```
with tf.variable_scope("input"):
...
with tf.variable_scope("domain"):
...

```
通过脚本载入模型，查看网络结构名和参数：
```
[<tf.Variable 'input/Variable:0' shape=(7540,) dtype=string_ref>, <tf.Variable 'input/Variable_1:0' shape=(7541, 300) dtype=float32_ref>, 
<tf.Variable 'domain/lstm_fused_cell/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'domain/lstm_fused_cell/bias:0' shape=(400,) dtype=float32_ref>, 
<tf.Variable 'domain/lstm_fused_cell_1/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'domain/lstm_fused_cell_1/bias:0' shape=(400,) dtype=float32_ref>, 
<tf.Variable 'domain/dense/kernel:0' shape=(200, 349) dtype=float32_ref>, <tf.Variable 'domain/dense/bias:0' shape=(349,) dtype=float32_ref>, 
<tf.Variable 'domain/crf:0' shape=(349, 349) dtype=float32_ref>, <tf.Variable 'domain/Variable:0' shape=(349,) dtype=string_ref>]
```
### configure.yaml中增加可配置参数 warm_start_dir
```
# 不启用热启动，即为None的时候，模型从头开始训练，不读入大模型的checkpoint值
warm_start_dir: 

# 启用热启动，其中"/home/shf/conda/ner/seq2annotation/model_dir/BilstmCrfModel-64-0.001-None-15000"为大模型results生成的checkpoint的绝对路径
warm_start_dir: "/home/shf/conda/ner/seq2annotation/model_dir/BilstmCrfModel-64-0.001-None-15000"

```

### estimator训练中增加增加参数 warm_start_from

(ner_s2s.ner_estimator.train_model.py)中 tf.estimator.Estimator 函数增加可配置选项 warm_start_from
* if 对configure.yaml中增加可配置参数 warm_start_dir：的传入值进行判断：不为None，则warm_start_from=ws，否则 warm_start_from=None
* 如果 warm_start_from=None，则不启用热启动，模型从头开始训练，也不读入训练好的大模型的checkpoint值
* 如果 warm_start_from=ws，则启用热启动，通过 tf.estimator.WarmStartSettings 函数对热启动中checkpoint里网络结构层进行选择
```
    if config.get("warm_start_dir") is not None:    
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config.get("warm_start_dir"),
            # vars_to_warm_start='.*domain/lstm_fused_cell.*', # checkpoint 只读入所有的lstm层
            # vars_to_warm_start='^(?!.*dense)',     # checkpoint 读入除了dense的所有层
            # vars_to_warm_start='.*',           # checkpoint 读入所有的神经网络层   
            vars_to_warm_start=['input/Variable_1', 'domain/lstm_fused_cell'],   # checkpoint 读入输入的embedding层与所有的lstm层      
            var_name_to_vocab_info=None,
            var_name_to_prev_var_name=None)
        estimator = tf.estimator.Estimator(
            model_fn, instance_model_dir, cfg, estimator_params, warm_start_from=ws
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn, instance_model_dir, cfg, estimator_params, warm_start_from=None
        )

```
