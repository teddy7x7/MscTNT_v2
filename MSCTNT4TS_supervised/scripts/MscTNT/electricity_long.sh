if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1440
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity_seq1440
data_name=custom

random_seed=2021
for pred_len in 720
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
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 6\
      --inner_mlp_ratio 4\
      --pos_drop 0\
      --head_dropout 0.05\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.05 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 360\
      --patch_stride 360\
      --subpatch_len 60 \
      --subpatch_stride 60 \
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 6\
      --early_outer_tcn_layers 6\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done