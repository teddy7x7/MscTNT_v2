if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=540
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=ETTh1.csv

model_id_name=ETTh1_Msc_diff_exShortLookback2

data_name=ETTh1

random_seed=2021

# for seq_len in 96 144 192 336 512 672 720
# for seq_len in 96 
# for seq_len in 336 512 672 720
for seq_len in 48 60 72 84 96
# for seq_len in 840 960 1080 1200
do
for pred_len in 96 192 336 720
# for pred_len in 192 336
# for pred_len in 96 
# for pred_len in 192
# for pred_len in 336
# for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name"_"od$outer_dim \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 4 \
      --outer_n_heads 4\
      --inner_n_heads 1\
      --outer_dim 16\
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
      --patch_len 24\
      --patch_stride 24\
      --subpatch_len 4\
      --subpatch_stride 4\
      --des 'Exp' \
      --train_epochs 100\
      --patience 100 \
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
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done