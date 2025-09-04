if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=120
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather_0In3OutTcn_ExlongPatch
data_name=custom

random_seed=2021
# for pred_len in 96 192 336
for pred_len in 96
# for pred_len in 720
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
      --enc_in 21 \
      --e_layers 1 \
      --outer_n_heads 4 \
      --inner_n_heads 1 \
      --outer_dim 80\
      --inner_dim 24\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0.05\
      --head_dropout 0.15 \
      --inner_tcn_drop 0.15 \
      --outer_tcn_drop 0.25 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.25 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.25 \
      --patch_len 12\
      --patch_stride 12\
      --subpatch_len 3 \
      --subpatch_stride 3 \
      --des 'Exp' \
      --train_epochs 50\
      --patience 25 \
      --itr 1 --batch_size 128 --learning_rate 0.00008\
      --pct_start 0.2\
      --lradj 'TST'\
      --revin 1\
      --subtract_last 0 \
      --affine 0 \
      --early_inner_tcn_layers 0 \
      --early_outer_tcn_layers 3 \
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 11000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done