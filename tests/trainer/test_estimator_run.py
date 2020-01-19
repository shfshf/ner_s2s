import os

import pytest

from ner_s2s.utils import remove_content_in_dir, create_dir_if_needed


@pytest.mark.skip("tf crf don't work in tf 1.15")
def test_main(datadir):
    workshop_dir = datadir
    # clean result dir first
    result_dir = os.path.join(workshop_dir, "./results")
    for target_dir in [
        os.path.join(result_dir, i)
        for i in ["h5_model", "model_dir", "saved_model", "summary_log_dir"]
    ]:
        create_dir_if_needed(target_dir)
        remove_content_in_dir(target_dir)

    config_file = os.path.join(workshop_dir, "./configure.yaml")

    os.environ["_DEFAULT_CONFIG_FILE"] = config_file

    # set current working directory to file directory
    os.chdir(workshop_dir)

    # TODO(howl-anderson): wrap up the train function into function call
    # import seq2annotation.trainer.cli_keras
