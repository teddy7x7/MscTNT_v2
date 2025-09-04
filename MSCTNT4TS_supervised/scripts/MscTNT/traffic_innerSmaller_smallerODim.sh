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
model_id_name=traffic_final_innerSmaller_smallerODim_$seq_len_$pred_len"_"od$outer_dim"_"id$inner_dim"_"opl$patch_len"_"ipl$subpatch_len"_"oh$outer_n_heads"_"ih$inner_n_heads

data_name=custom

random_seed=2021
# for pred_len in 720 96 192 336 
# for pred_len in 96 720 
for pred_len in 720
do

# for outer_dim in 20 40 60 80
for outer_dim in 60 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 1 \
      --n_heads 1 \
      --outer_n_heads 20\
      --inner_n_heads 4\
      --outer_dim $outer_dim\
      --inner_dim 12 \
      --pos_drop 0\
      --head_dropout 0.05 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --inner_tcn_drop 0.15 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.1 \
      --outer_attn_dropout 0.1 \
      --inner_proj_dropout 0.1 \
      --outer_proj_dropout 0.1 \
      --patch_len 180\
      --patch_stride 180\
      --subpatch_len 45 \
      --subpatch_stride 45 \
      --des 'Exp' \
      --train_epochs 50\
      --patience 10 \
      --lradj 'TST'\
      --pct_start 0.3\
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 1\
      --inner_tcn_layers 1\
      --outer_tcn_layers 1\
      --final_div_factor 11000\
      --itr 1 --batch_size 32 --learning_rate 0.0003\
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done