export CUDA_VISIBLE_DEVICES=1

MODEL="ARTransformer" # the teacher model type 
DATASET="hourly" # dataset type 
for N_LAYER in 6 # Number of teacher transformer layers 
do 
for BSZ in 64 # batch size 
do
for seed in  1234 
do 
python3 ar_kd_teacher.py --model $MODEL  --num_layer ${N_LAYER} --seed ${seed} --dataset $DATASET \
         --log "teacher_ckpt/$DATASET-${MODEL}-${N_LAYER}L-seed-${seed}-bsz$BSZ-cor$COR.txt" \
         --batch_size $BSZ --max_epoch 20 --eval_step 200  --patience -1
echo $seed finished!
done
done 
done 
done 