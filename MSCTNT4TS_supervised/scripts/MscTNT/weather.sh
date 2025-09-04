if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=600
seq_len=720
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather_FIN_trySmaller3_diffPatchLen120
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720
# for pred_len in 192 336
# for pred_len in 96 720
# for pred_len in 720
for pred_len in 96
do

# for patch_len in 12 36 60 90 120 144
# for patch_len in 12 
# for patch_len in 60
# for patch_len in 90
for patch_len in 120
do

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name"_"$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --outer_n_heads 12\
      --inner_n_heads 1 \
      --outer_dim 36\
      --inner_dim 12 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0 \
      --inner_tcn_drop 0.15 \
      --outer_tcn_drop 0.2 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.05 \
      --outer_proj_dropout 0.05 \
      --patch_len $patch_len\
      --patch_stride $patch_len\
      --subpatch_len 6\
      --subpatch_stride 6\
      --des 'Exp' \
      --train_epochs 50\
      --patience 15\
      --itr 1 --batch_size 128 --learning_rate 0.0003\
      --pct_start 0.3\
      --lradj 'TST'\
      --revin 1\
      --subtract_last 1\
      --affine 0 \
      --early_inner_tcn_layers 1 \
      --early_outer_tcn_layers 1 \
      --inner_tcn_layers 1\
      --outer_tcn_layers 1\
      --final_div_factor 11000\
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_plen'$patch_len.log 
done
done