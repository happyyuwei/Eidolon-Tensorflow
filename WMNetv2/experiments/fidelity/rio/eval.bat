

cd ../

python main_script.py --data=../../../data/rio/train^
    --model=../../../trained_models/models/rio^
    --visual_result=./rio/results/baseline

python main_script.py --data=../../../data/rio/train^
    --model=../../../trained_models/models/rio^
    --watermark=../../watermark/wm_binary_x32.png^
    --wm_width=32^
    --is_binary=True^
    --visual_result=./rio/results/wm_b

python main_script.py --data=../../../data/rio/train^
    --model=../../../trained_models/models/rio^
    --watermark=../../watermark/wm_x32.png^
    --wm_width=32^
    --is_binary=False^
    --visual_result=./rio/results/wm_c

@pause