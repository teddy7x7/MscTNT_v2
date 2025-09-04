if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=540
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1_FIN_2diffPatchLen_forplot

data_name=ETTh1
# data_name=custom

random_seed=2021

for patch_len in 270 18 30 60 90 180
# for patch_len in 30
# for patch_len in 60
# for patch_len in 90
# for patch_len in 12 60 180 
do
# for seq_len in 96 144 192 336 512 672 720
# for seq_len in 96 144 192 336 512 672 720
# do

# for shuffle in "RandShuf" "HalfEx"
# do
# for pred_len in 96 192 336 720
# for pred_len in 720 192 336
# for pred_len in 192 336
# for pred_len in 96 
# for pred_len in 192
# for pred_len in 336
for pred_len in 720
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
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --outer_n_heads 4\
      --inner_n_heads 1\
      --outer_dim 16\
      --inner_dim 1 \
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0.05 \
      --head_dropout 0.05\
      --inner_tcn_drop 0.1 \
      --outer_tcn_drop 0.1 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len $patch_len\
      --patch_stride $patch_len\
      --subpatch_len 15\
      --subpatch_stride 15\
      --des 'Exp' \
      --train_epochs 1\
      --patience 100 \
      --itr 1 --batch_size 128 --learning_rate 0.0001\
      --pct_start 0.3\
      --lradj 'TST'\
      --revin 1\
      --subtract_last 0\
      --affine 0\
      --early_inner_tcn_layers 0\
      --early_outer_tcn_layers 0\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 11000\
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_plen'$patch_len.log
done
done
# done