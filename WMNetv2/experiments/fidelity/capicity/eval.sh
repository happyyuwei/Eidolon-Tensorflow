
cd ../

# python main_script.py --data=../../../data/flower/train \
#     --model=../../../app/flower_unet_cw_x32/model \
#     --visual_result=./capicity/results/baseline

python main_script.py --data=../../../data/flower/train \
    --model=../../../app/flower_unet_cw_x64/model \
    --watermark=../../watermark/wm_x64.png \
    --wm_width=64 \
    --is_binary=False \
    --visual_result=./capicity/results/wm_x64
