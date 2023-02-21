#python ../test_models.py era \
#    --weights=../checkpoint/TSM_era_RGB_pyconv_avg_segment1_e30/ckpt.best.pth.tar \
#    --test_segments=1 --batch_size=64 -j 4 --test_crops=3 \
#    --mode s1 --energy_thr 0.8

#python ../test_models.py era \
#    --weights=../checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s1_68/ckpt.best.pth.tar \
#    --test_segments=16 --batch_size=10 -j 4 --test_crops=3 \
#    --mode s1 --VAP

#python ../test_models.py era \
#    --weights=./checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s2/ckpt.best.pth.tar \
#    --test_segments=16 --batch_size=10 -j 4 --test_crops=1 \
#    --mode s2 --VAP

#python ../test_models.py era \
#    --weights=./checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s2/ckpt.best.pth.tar \
#    --test_segments=16 --batch_size=20 -j 4 --test_crops=1 \
#    --mode s2 --VAP

python ../test_models.py mod20 \
    --weights=./checkpoint/TSM_mod20_RGB_pyconv_avg_segment16_e40_VAP_s2/ckpt.best.pth.tar \
    --test_segments=16 --batch_size=20 -j 4 --test_crops=1 \
    --mode s2 --VAP

#python ../test_models.py mod20 \
#    --weights=./checkpoint/TSM_mod20_RGB_resnet50_avg_segment16_e30_VAP_s1/ckpt.pth.tar \
#    --test_segments=16 --batch_size=20 -j 4 --test_crops=1 \
#    --mode s1 --VAP

#python ../test_models.py mod20 \
#    --weights=./checkpoint/TSM_mod20_RGB_pyconv_avg_segment16_e40_VAP_s2/ckpt.best.pth.tar \
#    --test_segments=16 --batch_size=20 -j 4 --test_crops=1 \
#    --mode s2 --VAP

#python ../test_models.py action \
#    --weights=./checkpoint/TSM_action_RGB_pyconv_avg_segment16_e150_dense_VAP_s1/ckpt.best.pth.tar \
#    --test_segments=16 --batch_size=10 -j 4 --test_crops=10 \
#    --mode s1 --VAP