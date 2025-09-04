if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=720
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity_smallerDim_$seq_len_$pred_len"_"od$outer_dim"_"id$inner_dim"_"opl$patch_len"_"ipl$subpatch_len"_"oh$outer_n_heads"_"ih$inner_n_heads
data_name=custom

random_seed=2021
for pred_len in 720
do
# for outer_dim in 80 # original dim
for outer_dim in 10 20 40 60

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 720 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 180 \
      --patch_stride 180 \
      --subpatch_len 30 \
      --subpatch_stride 30 \
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 24\
      --patch_stride 24\
      --subpatch_len 4\
      --subpatch_stride 4\
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

      python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 144 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 36\
      --patch_stride 36\
      --subpatch_len 6\
      --subpatch_stride 6\
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

      python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 192 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 48\
      --patch_stride 48\
      --subpatch_len 8\
      --subpatch_stride 8\
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

      python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 84\
      --patch_stride 84\
      --subpatch_len 14\
      --subpatch_stride 14\
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

      python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 512 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 128\
      --patch_stride 128\
      --subpatch_len 32 \
      --subpatch_stride 32 \
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

      python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 672 \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --outer_n_heads 10 \
      --inner_n_heads 10\
      --outer_dim 80\
      --inner_dim 40\
      --outer_mlp_ratio 1\
      --inner_mlp_ratio 1\
      --pos_drop 0\
      --head_dropout 0\
      --inner_tcn_drop 0.05 \
      --outer_tcn_drop 0.15 \
      --inner_attn_dropout 0.15 \
      --outer_attn_dropout 0.15 \
      --inner_proj_dropout 0.15 \
      --outer_proj_dropout 0.15 \
      --patch_len 168\
      --patch_stride 168\
      --subpatch_len 28 \
      --subpatch_stride 28 \
      --des 'Exp' \
      --train_epochs 40\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --revin 1\
      --subtract_last 1\
      --affine 0\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 3\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 10000\
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done