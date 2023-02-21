#python ../main.py era RGB \
#     --arch pyconv --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 12 24 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 --npb \
#     --mode s2 --energy_thr 0.6 \
#     --tune_from=../checkpoint/TSM_era_RGB_pyconv_avg_segment8_e30/ckpt.best.pth.tar
#     --shift --shift_div=8 --shift_place=blockres

#python ../main.py era RGB \
#     --arch pyconv --num_segments 16 \
#     --gd 20 --lr 0.0005 --lr_steps 12 24 --epochs 40 \
#     --batch-size 64 -j 4 --consensus_type=avg --dropout 0.8 --eval-freq=1 --npb --wd 1e-4 \
#     --mode s2  --VAP \
#     --tune_from=./checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s1/ckpt.best.pth.tar

#python ../main.py era RGB \
#     --arch pyconv --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 12 24 --epochs 40 \
#     --batch-size 64 -j 4 --consensus_type=avg --dropout 0.8 --eval-freq=1 --npb --wd 1e-4 \
#     --mode s2  --VAP \
#     --tune_from=./checkpoint/TSM_era_RGB_pyconv_avg_segment8_e40_VAP_s1/ckpt.best.pth.tar


#python ../main.py mod20 RGB \
#     --arch pyconv --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 12 24 --epochs 40 \
#     --batch-size 64 -j 4 --consensus_type=avg --dropout 0.8 --eval-freq=1 --npb --wd 5e-4 \
#     --mode s2 --VAP \
#     --tune_from=./checkpoint/TSM_mod20_RGB_pyconv_avg_segment8_e40_VAP_s1/ckpt.best.pth.tar


python ../main.py mod20 RGB \
     --arch pyconv --num_segments 16 \
     --gd 20 --lr 0.001 --lr_steps 12 24 --epochs 40 \
     --batch-size 32 -j 4 --consensus_type=avg --dropout 0.5 --eval-freq=1 --npb --wd 1e-4 \
     --mode s2  --VAP \
     --tune_from=./checkpoint/TSM_mod20_RGB_pyconv_avg_segment16_e40_VAP_s1/ckpt.best.pth.tar


#python ../main.py action RGB \
#     --arch pyconv --num_segments 16 \
#     --gd 20 --lr 0.003 --lr_steps 48 96 --epochs 120 \
#     --batch-size 32 -j 4 --consensus_type=avg --dropout 0.3 --eval-freq=4 --npb --wd 5e-4 \
#     --mode s1 --VAP --dense_sample \
#     --tune_from=./checkpoint/TSM_action_RGB_pyconv_avg_segment16_e120_dense_VAP_s1/ckpt.best.pth.tar