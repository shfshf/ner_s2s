import copy
import os

import tensorflow as tf

from ner_s2s import utils
from ner_s2s.utils import class_from_module_path
from tokenizer_tools.hooks import TensorObserveHook


tf.logging.set_verbosity(tf.logging.INFO)

observer_hook = TensorObserveHook(
    {"fake_golden": "fake_golden:0", "fake_prediction": "fake_prediction:0"},
    {
        # "word_str": "word_strings_Lookup:0",
        "predictions_id": "predictions:0",
        "predict_str": "predict_Lookup:0",
        "labels_id": "labels:0",
        "labels_str": "IteratorGetNext:2",
    },
    {
        "word_str": lambda x: x.decode(),
        "predict_str": lambda x: x.decode(),
        "labels_str": lambda x: x.decode(),
    },
)


def train_model(train_inpf, eval_inpf, config, model_fn, model_name):
    estimator_params = copy.deepcopy(config)

    indices = [idx for idx, tag in enumerate(config["tags_data"]) if tag.strip() != "O"]
    num_tags = len(indices) + 1
    estimator_params["_indices"] = indices
    estimator_params["_num_tags"] = num_tags

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=config["save_checkpoints_secs"])

    model_specific_name = "{model_name}-{batch_size}-{learning_rate}-{max_steps}-{max_steps_without_increase}".format(
        model_name=model_name,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        max_steps_without_increase=config["max_steps_without_increase"],
    )

    instance_model_dir = os.path.join(config["model_dir"], model_specific_name)

    warm_start_setting = None
    if config.get("warm_start_dir"):
        warm_start_setting = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config.get("warm_start_dir"),
            vars_to_warm_start=[
                'task_independent/Variable_1',
                'task_independent/lstm_fused_cell'
            ],
            # vars_to_warm_start='.*'  # all warm_start
        )
    estimator = tf.estimator.Estimator(
        model_fn, instance_model_dir, cfg, estimator_params, warm_start_from=warm_start_setting
    )

    # Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    utils.create_dir_if_needed(estimator.eval_dir())

    # build hooks from config
    train_hook = []
    if config.get("early_stop"):
        hook_params = config.get('hook')['stop_if_no_increase']
        hook_train = tf.estimator.experimental.stop_if_no_increase_hook(
            estimator,
            'f1',
            max_steps_without_increase=hook_params['max_steps_without_increase'],
            min_steps=hook_params['min_steps'],
            run_every_secs=hook_params['run_every_secs']
        )
        train_hook = [hook_train]
    else:
        for i in config.get("train_hook", []):
            class_ = class_from_module_path(i["class"])
            params = i["params"]
            if i.get("inject_whole_config", False):
                params["config"] = config
            train_hook.append(class_(**params))

    eval_hook = []
    for i in config.get("eval_hook", []):
        class_ = class_from_module_path(i["class"])
        params = i["params"]
        if i.get("inject_whole_config", False):
            params["config"] = config
        eval_hook.append(class_(**params))

    if eval_inpf:
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_inpf, hooks=train_hook, max_steps=config["max_steps"]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_inpf, throttle_secs=config["throttle_secs"], hooks=eval_hook
        )
        evaluate_result, export_results = tf.estimator.train_and_evaluate(
            estimator, train_spec, eval_spec
        )
    else:
        estimator.train(
            input_fn=train_inpf, hooks=train_hook, max_steps=config["max_steps"]
        )
        evaluate_result, export_results = {}, None

    # export saved_model
    feature_spec = {
        # 'words': tf.placeholder(tf.int32, [None, None]),
        "words": tf.placeholder(tf.string, [None, None]),
        "words_len": tf.placeholder(tf.int32, [None]),
    }

    if config.get("forced_saved_model_dir"):
        instance_saved_dir = config.get("forced_saved_model_dir")
    else:
        instance_saved_dir = os.path.join(
            config["saved_model_dir"], model_specific_name
        )

    utils.create_dir_if_needed(instance_saved_dir)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_spec
    )
    raw_final_saved_model = estimator.export_saved_model(
        instance_saved_dir,
        serving_input_receiver_fn,
        # assets_extra={
        #     'tags.txt': 'data/tags.txt',
        #     'vocab.txt': 'data/unicode_char_list.txt'
        # }
    )

    final_saved_model = raw_final_saved_model.decode("utf-8")

    return evaluate_result, export_results, final_saved_model
