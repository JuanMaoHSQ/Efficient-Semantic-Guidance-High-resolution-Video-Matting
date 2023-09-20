#!/usr/bin/env bash
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "checkpoint/stage3/epoch-27.pth" \
    --device cuda \
    --input-source "videoeva/1.mov" \
    --output-type video \
    --output-composition "videoeva/COM1.mov" \
    --output-video-mbps 4
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "checkpoint/stage3/epoch-27.pth" \
    --device cuda \
    --input-source "videoeva/2.mov" \
    --output-type video \
    --output-composition "videoeva/COM2.mov" \
    --output-video-mbps 4
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "checkpoint/stage3/epoch-27.pth" \
    --device cuda \
    --input-source "videoeva/3.mov" \
    --output-type video \
    --output-composition "videoeva/COM3.mov" \
    --output-video-mbps 4


