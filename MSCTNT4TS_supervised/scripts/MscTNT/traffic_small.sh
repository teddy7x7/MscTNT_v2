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
model_id_name=traffic_noInTCN_small2
data_name=custom

random_seed=2021
for pred_len in 96
# for pred_len in 96 192 336 720
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
      --n_heads 4 \
      --outer_n_heads 20\
      --inner_n_heads 4\
      --outer_dim 60\
      --inner_dim 40 \
      --pos_drop 0\
      --head_dropout 0 \
      --inner_tcn_drop 0.1 \
      --outer_tcn_drop 0.1 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.05 \
      --outer_proj_dropout 0.05 \
      --patch_len 120\
      --patch_stride 120\
      --subpatch_len 30 \
      --subpatch_stride 30 \
      --des 'Exp' \
      --train_epochs 30\
      --patience 6\
      --lradj 'TST'\
      --pct_start 0.3\
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --revin_eps 1e-5\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 1\
      --outer_tcn_layers 1\
      --final_div_factor 11000\
      --itr 1 --batch_size 32 --learning_rate 0.0003\
      --reduce_dim_tcn_channel_list '862'\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done