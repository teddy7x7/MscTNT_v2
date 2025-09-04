if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1_Final
data_name=ETTm1

random_seed=2021
# for pred_len in 192 336 720
for pred_len in 96 
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
      --enc_in 7 \
      --e_layers 1 \
      --outer_n_heads 12\
      --inner_n_heads 1 \
      --outer_dim 12 \
      --inner_dim 12 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0.05\
      --head_dropout 0.05\
      --inner_tcn_drop 0.2 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 180\
      --patch_stride 180\
      --subpatch_len 45\
      --subpatch_stride 45\
      --des 'Exp' \
      --train_epochs 100\
      --patience 30\
      --itr 1 --batch_size 128 --learning_rate 0.0006\
      --pct_start 0.3\
      --lradj 'TST'\
      --revin 1\
      --subtract_last 0\
      --affine 0\
      --early_inner_tcn_layers 3\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done