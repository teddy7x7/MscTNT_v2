if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=600
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2_Uni
data_name=ETTh2

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --outer_n_heads 4\
      --inner_n_heads 1\
      --outer_dim 12\
      --inner_dim 4 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0 \
      --head_dropout 0.05\
      --inner_tcn_drop 0.1 \
      --outer_tcn_drop 0.1 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.1 \
      --outer_proj_dropout 0.1 \
      --patch_len 120\
      --patch_stride 120\
      --subpatch_len 24\
      --subpatch_stride 24\
      --des 'Exp' \
      --train_epochs 40\
      --patience 10 \
      --itr 1 --batch_size 128 --learning_rate 0.0001\
      --pct_start 0.3\
      --lradj 'TST'\
      --revin 1\
      --subtract_last 0\
      --affine 0\
      --early_inner_tcn_layers 0\
      --early_outer_tcn_layers 1\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 11000\
      --inner_repatching False \
      --outer_repatching True \
       | tee -a logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log 
done