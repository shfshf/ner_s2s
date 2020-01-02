from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from ner_s2s.ner_estimator.input import build_input_func, generate_tagset
from ner_s2s.ner_estimator.model import Model

from typing import Any
from deliverable_model.request import Request
from deliverable_model.response import Response
from deliverable_model.utils import create_dir_if_needed
from ner_s2s.deliver_utils import export_as_deliverable_model

from tensorflow.python.saved_model import tag_constants
import mlflow
import mlflow.tensorflow


# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()


def converter_for_request(request: Request) -> Any:
    from micro_toolkit.data_process.text_sequence_padding import TextSequencePadding

    tsp = TextSequencePadding('<pad>')
    return {
        "words": tsp.fit(request.query),
        "words_len": [
            len(list(filter(lambda x: x != 0.0, text))) for text in request.query
        ],
    }


def converter_for_response(response: Any) -> Response:
    from deliverable_model.response import Response

    return Response(response["tags"])


def main():
    with mlflow.start_run():
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

        #  mlflow metrics
        # mlflow.log_metrics(evaluate_result, step=100)

        # mlflow Logging the saved model
        mlflow.tensorflow.log_model(tf_saved_model_dir=final_saved_model,
                                    tf_meta_graph_tags=[tag_constants.SERVING],
                                    tf_signature_def_key="serving_default",
                                    artifact_path="model")

        export_as_deliverable_model(
            create_dir_if_needed(config["deliverable_model_dir"]),
            tensorflow_saved_model=final_saved_model,
            converter_for_request=converter_for_request,
            converter_for_response=converter_for_response,
            addition_model_dependency=["micro_toolkit"]
        )


if __name__ == "__main__":
    main()
