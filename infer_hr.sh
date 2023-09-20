#!/usr/bin/env bash
python inference_speed_test.py \
    --model-variant mobilenetv3 \
    --resolution 512 288 \
    --downsample-ratio 1 \
    --precision float32
