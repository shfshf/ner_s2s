#!/usr/bin/env bash
python3 -m ner_s2s.ner_keras.keras_run

chown -R `stat -c "%u:%g" /data` /data