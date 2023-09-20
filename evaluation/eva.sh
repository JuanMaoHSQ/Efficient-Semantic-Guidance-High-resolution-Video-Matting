#!/usr/bin/env bash
python evaluate_hr.py \
    --pred-dir evaltest/pre_s3ep27 \
    --true-dir evaltest/gd
    --num-workers 48



