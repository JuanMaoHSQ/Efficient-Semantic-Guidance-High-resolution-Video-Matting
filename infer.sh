#!/usr/bin/env bash
check="checkpoint/stage4/epoch-35.pth"
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0000/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0000/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0000/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0000/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0000/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0000/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0000/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0000/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0001/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0001/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0001/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0001/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0001/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0001/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0001/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0001/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0002/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0002/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0002/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0002/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0002/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0002/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0002/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0002/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0003/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0003/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0003/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0003/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0003/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0003/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0003/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0003/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0004/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0004/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0004/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0004/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0004/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0004/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0004/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0004/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0005/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0005/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0005/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0005/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0005/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0005/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0005/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0005/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0006/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0006/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0006/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0006/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0006/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0006/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0006/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0006/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0007/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0007/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0007/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0007/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0007/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0007/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0007/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0007/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0008/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0008/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0008/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0008/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0008/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0008/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0008/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0008/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0009/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0009/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0009/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0009/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0009/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0009/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0009/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0009/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0010/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0010/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0010/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0010/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0010/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0010/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0010/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0010/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0011/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0011/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0011/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0011/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0011/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0011/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0011/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0011/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0012/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0012/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0012/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0012/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0012/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0012/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0012/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0012/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0013/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0013/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0013/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0013/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0013/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0013/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0013/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0013/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0014/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0014/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0014/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0014/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0014/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0014/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0014/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0014/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0010/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0010/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0010/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0010/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0015/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0015/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0015/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0015/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0016/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0016/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0016/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0016/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0016/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0016/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0016/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0016/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0017/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0017/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0017/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0017/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0017/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0017/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0017/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0017/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0018/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0018/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0018/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0018/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0018/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0018/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0018/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0018/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0019/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0019/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0019/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0019/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0019/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0019/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0019/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0019/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0020/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0020/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0020/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0020/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0020/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0020/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0020/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0020/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0021/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0021/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0021/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0021/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0021/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0021/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0021/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0021/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0022/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0022/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0022/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0022/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0022/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0022/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0022/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0022/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0023/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0023/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0023/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0023/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0023/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0023/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0023/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0023/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_static/0024/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_static/0024/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_static/0024/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_static/0024/fgr" \
    --seq-chunk 12 \
    --num-workers 8
python inference.py \
    --variant mobilenetv3 \
    --checkpoint $check \
    --device cuda \
    --input-source "evaluation/evaltest/gd/videomatte_motion/0024/com" \
    --output-type png_sequence \
    --output-composition "evaluation/evaltest/pre/videomatte_motion/0024/com" \
    --output-alpha "evaluation/evaltest/pre/videomatte_motion/0024/pha" \
    --output-foreground "evaluation/evaltest/pre/videomatte_motion/0024/fgr" \
    --seq-chunk 12 \
    --num-workers 8
