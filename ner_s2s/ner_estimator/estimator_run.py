from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from ner_s2s.input import build_input_func, generate_tagset
from ner_s2s.ner_estimator.model import Model
from ner_s2s.ner_estimator.estimator_utils import export_as_deliverable_model

from deliverable_model.utils import create_dir_if_needed
from deliverable_model.converter_base import ConverterBase
from seq2annotation_for_deliverable.main import (
    ConverterForRequest,
    ConverterForResponse,
)

from tensorflow.python.saved_model import tag_constants
import mlflow
import mlflow.tensorflow


# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()


def main():

    raw_config = read_configure()
    model = Model(raw_config)

    config = model.get_default_config()  # 默认的配置
    config.update(raw_config)

    #  mlflow hyperparameter
    mlflow.log_param("Batch_Size", config["batch_size"])
    mlflow.log_param("Learning_Rate", config["learning_rate"])
    mlflow.log_param("Epochs", config["epochs"])
    mlflow.log_param("Embedding_Dim", config["embedding_dim"])

    corpus = get_corpus_processor(config)
    corpus.prepare()
    train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
    eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

    corpus_meta_data = corpus.get_meta_info()

    config["tags_data"] = generate_tagset(corpus_meta_data["tags"])

    # train and evaluate model
    train_input_func = build_input_func(train_data_generator_func, config)
    eval_input_func = (
        build_input_func(eval_data_generator_func, config)
        if eval_data_generator_func
        else None
    )

    evaluate_result, export_results, final_saved_model = model.train_and_eval_then_save(
        train_input_func, eval_input_func, config
    )

    # mlflow metrics
    # mlflow.log_metrics(evaluate_result, step=100)

    # mlflow Logging the saved model
    mlflow.tensorflow.log_model(tf_saved_model_dir=final_saved_model,
                                tf_meta_graph_tags=[tag_constants.SERVING],
                                tf_signature_def_key='serving_default',
                                artifact_path='model')

    export_as_deliverable_model(
        create_dir_if_needed(config["deliverable_model_dir"]),
        tensorflow_saved_model=final_saved_model,
        converter_for_request=ConverterForRequest(),
        converter_for_response=ConverterForResponse(),
        addition_model_dependency=["micro_toolkit", "seq2annotation_for_deliverable"],
    )


if __name__ == "__main__":
    main()
