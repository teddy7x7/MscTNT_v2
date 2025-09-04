if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic_final_diffPatchLen3_diffSmallestPatch
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720 144 512 672
# for pred_len in 144 
# for pred_len in 96 720
for pred_len in 720 96  
# for pred_len in 96 
do

# for patch_len in 12
# for patch_len in 60 120 240 360
for patch_len in 36
# for patch_len in 60
# for patch_len in 120 
# for patch_len in 240
# for patch_len in 360
do

# for subpatch_len in 1 3 4 6
for subpatch_len in 3 9 12 18
# for subpatch_len in 5 15 20 30
# for subpatch_len in 10 30 40 60
# for subpatch_len in 20 60 80 120
# for subpatch_len in 30 90 120 180
do 

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name"_"$seq_len'_'$pred_len'_pthLen'$patch_len'_spthLen'$subpatch_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 1 \
      --n_heads 1 \
      --outer_n_heads 40\
      --inner_n_heads 12\
      --outer_dim 40\
      --inner_dim 24 \
      --pos_drop 0\
      --head_dropout 0 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --inner_tcn_drop 0.1 \
      --outer_tcn_drop 0.1 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.05 \
      --outer_proj_dropout 0.05 \
      --patch_len $patch_len\
      --patch_stride $patch_len\
      --subpatch_len $subpatch_len \
      --subpatch_stride $subpatch_len \
      --des 'Exp' \
      --train_epochs 50\
      --patience 10 \
      --lradj 'TST'\
      --pct_start 0.3\
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 0\
      --early_outer_tcn_layers 0\
      --inner_tcn_layers 1\
      --outer_tcn_layers 1\
      --final_div_factor 11000\
      --itr 1 --batch_size 32 --learning_rate 0.0003\
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_plen'$patch_len'_splen'$subpatch_len.log 
done
done
done