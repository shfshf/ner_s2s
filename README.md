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
* Data source 部分指定了项目文件的数据（data）的相对路径：包括训练集、测试集、实体、字典等；
* model configure 部分指定了一些常用模型的超参数配置（包括estimator与keras模式），例如batch_size，epochs，learning_rate等
* Data output 部分指定了模型生成的多种保存格式的相对路径，包括pb，h5等，其中deliverable_model对模型进行了前置，后置处理，推理、解码进行了封装
## ner_s2s 训练及推理命令

### ner_s2s 训练
### cpu (没有GPU加速)
```
# estimator 模式
python -m ner_s2s.ner_estimator.estimator_run
# keras 模式
python -m ner_s2s.ner_keras.keras_run
```
### gpu 指定显卡运行(针对多显卡)
```
# estimator 模式
CUDA_VISIBLE_DEVICES=0 python -m ner_s2s.ner_estimator.estimator_run
# keras 模式
CUDA_VISIBLE_DEVICES=1 python -m ner_s2s.ner_keras.keras_run
```
### 

## 制作docker镜像 
### Makefile
docker打包镜像的入口命令在根目录文件夹下的 Makefile 文件中，其中：
```
.PHONY: build_docker_nightly_build
build_docker_nightly_build: build_docker_nightly_build_trainer build_docker_nightly_build_server

.PHONY: build_docker_nightly_build_trainer
build_docker_nightly_build_trainer: dist
	cp -r dist docker/nightly/trainer/
	docker rmi -f ner_trainer
	docker build --no-cache --force-rm --tag ner_trainer --file docker/nightly/trainer/Dockerfile docker_v2/nightly/trainer/

.PHONY: build_docker_nightly_build_server
build_docker_nightly_build_server: dist
	cp -r dist docker/nightly/server/
	docker rmi -f ner_server
	docker build --no-cache --force-rm --tag ner_server --file docker/nightly/server/Dockerfile docker_v2/nightly/server/
```
首先要开启docker服务，然后在terminal中输入命令（在根目录下）：
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
	docker build --no-cache --force-rm --tag shf_ner_trainer --file docker/nightly/trainer/Dockerfile docker_v2/nightly/trainer/
```
其他make命令无需改变。

执行结果：
```
shf@tesla:~$ docker images
REPOSITORY         TAG       IMAGE ID         CREATED          SIZE
ner_server        latest    be9fa79c6565     9 seconds ago     4.4GB
shf_ner_trainer   latest    545cd33fa016     2 minutes ago     4.4GB
```
## 测试生成的docker镜像 
新建名为test文件夹里面新建三个文件：
* data文件夹：里面存放数据集，与上面的ner_s2s项目中data文件夹的内容是相似的，但是需要加入一个新的名为ucloud_configure.json的配置文件夹；
* output文件夹：用来保存训练模型的结果；
* run_docker.bash: 用来启动docker命令。

接下来进一步介绍 ucloud_configure.json 与 run_docker.bash
### run_docker.bash
terminal启动命令(当前文件目录下):
```
bash run_docker.bash
```
run_docker.bash 内容:
```
nvidia-docker run -it --rm -v /home/shf/test/docker/ner_s2s/data:/data/data -v /home/shf/test/docker/ner_s2s/output:/data/output shf_ner_trainer:latest /bin/bash -c "cd /data && 
/usr/bin/python3 -m ner_s2s.ner_estimator.estimator_run --ioflow_default_configure=/data/data/ucloud_configure.json --num_gpus=1 -- work_dir=/data --data_dir=/data/data --output_dir=/data/output"
```
其中：
* /home/shf/test/docker/ner_s2s/data      ---存放data文件夹的绝对路径
* /home/shf/test/docker/ner_s2s/output    ---存放output输出文件的绝对路径
* shf_ner_trainer:latest                         ---镜像名：版本号
* -m ner_s2s.ner_estimator.estimator_run         ---启动ner_s2s项目的命令
* --ioflow_default_configure=/data/data/ucloud_configure.json   ---指定ucloud_configure.json配置文件的路径

### ucloud_configure.json
```
{
  "use_tpu": false,
  "data_source_scheme": "local",
  "train": "/data/data/train.conllx",
  "test": "/data/data/test.conllx",
  "tags": "/data/data/entity.txt",
  "vocabulary_file": "/data/data/unicode_char_list.txt",
  "constraint": "/data/data/constraint.json",
  "intent_field": "domain",
  "shuffle_pool_size": 1000,
  "dropout": 0.5,
  "batch_size": 64,
  "epochs": 35,
  "max_steps": null,
  "max_steps_without_increase": 15000,
  "embedding_vocabulary_size": 7540,
  "embedding_dim": 300,
  "lstm_size": 100,
  "max_sentence_len": 45,
  "bilstm_stack_config":[
     {
      "units": 100, 
      "dropout": 0.5, 
      "recurrent_dropout": 0.5
     }
  ],
  "result_dir": "/data/output",
  "params_log_file": "/data/output/params.json",
  "model_dir": "/data/output/model_dir",
  "h5_model_file": "/data/output/h5_model/model.h5",
  "saved_model_dir": "/data/output/saved_model",
  "deliverable_model_dir": "/data/output/deliverable_model",
  "summary_log_dir": "/data/output/summary_log_dir",
  "save_checkpoints_secs": 60,
  "throttle_secs": 60,
  "tpu_name": "",
  "tpu_zone": "",
  "gcp_project": "",
  "task_id": "ABC123"
}
```
其中：
* "train": "/data/data/train.conllx", 指定了训练集的相对路径
* "test": "/data/data/test.conllx", 指定了测试集的相对路径
* "tags": "/data/data/entity.txt", 指定了实体标签的相对路径
* "vocabulary_file": "/data/data/unicode_char_list.txt", 指定了字典的相对路径
* "dropout": 0.5, estimator模式下dropout值的超参数
* "batch_size": 64, 模型网络结构的batch_size值的超参数，即1个batch包含的样本数目，通常为2的n次幂
* "epochs": 35, 模型数据集重复训练的次数epoch值的超参数
* "embedding_vocabulary_size": 7540, 指定字典大小的值
* "max_sentence_len": 45, 指定句子最长大小的值
* "bilstm_stack_config": keras模式下，bilstim里面的超参数配置
* "model_dir": "/data/output/model_dir", 保存模型的checkpoint文件，可以通过tensorboard查看模型指标的变化情况
* "h5_model_file": "/data/output/h5_model/model.h5", keras模式下，会保存模型的h5格式
* "saved_model_dir": "/data/output/saved_model", 保存模型的pb格式的文件
* "deliverable_model_dir": "/data/output/deliverable_model", 对模型进行了前置，后置处理，推理、解码进行了封装
### ner_s2s 推理
...


