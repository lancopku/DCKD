TIME="hourly" # five_minute
TEACHER_MODEL=teacher_ckpt/$PATH_TO_TEACHER_MODEL #hourly-ARTransformer-6L-baseline-seed-98-bsz32-cor0.0.pt 
for AR in gs # deep ar type, gs for Gaussian 
do
for N_LAYER in 1  # number of student layers 
do 
for BSZ in 32 # batch size 
do
for KDW in 1.0 # Distribution KD loss
do
for AKDW in 5.0 # angle loss weight
do
for RKDW in 5.0 #  distance loss weight 
do 
for DATA_RATIO in 1.0 # full dataset, change to portion like 0.1 for low-resource settings 
do 
for seed in 1 2 3 4 5 6 7 # seed  
do
python3 ar_kd_main.py --patience 10  --model $MODEL --ar $AR  --data_ratio $DATA_RATIO --num_layer ${N_LAYER} --seed ${seed} --dataset $TIME \
    --log "log/$TIME-${MODEL}-${N_LAYER}L-deepAR-kd-seed-${seed}-${AR}-bsz$BSZ-KDW$KDW-RKDW$RKDW-jf-AKDW$AKDW-abs-dataR${DATA_RATIO}.txt" \ 
    --max_epoch 20  --batch_size $BSZ --eval_step 1000  --teacher_path $TEACHER_MODEL --kd_loss_w $KDW --akd_loss_w $AKDW  --rkd_loss_w $RKDW 
echo $seed finished!
done 
done
done 
done 
done 
done 
done 
done 