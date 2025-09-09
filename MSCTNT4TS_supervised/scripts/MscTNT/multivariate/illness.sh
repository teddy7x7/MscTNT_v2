if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=120
model_name=MscTNT4TS

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness_FIN
data_name=custom

random_seed=2021
for pred_len in 24 36 48 60
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
      --outer_n_heads 6 \
      --inner_n_heads 1 \
      --outer_dim 12 \
      --inner_dim 5 \
      --outer_mlp_ratio 4\
      --inner_mlp_ratio 2\
      --pos_drop 0\
      --head_dropout 0.05\
      --inner_tcn_drop 0.15 \
      --outer_tcn_drop 0.05 \
      --inner_attn_dropout 0.05 \
      --outer_attn_dropout 0.05 \
      --inner_proj_dropout 0.25 \
      --outer_proj_dropout 0.25 \
      --patch_len 12\
      --patch_stride 12\
      --subpatch_len 3\
      --subpatch_stride 3\
      --des 'Exp' \
      --train_epochs 100\
      --patience 100\
      --lradj 'TST' \
      --itr 1 --batch_size 16 --learning_rate 0.0025 \
      --pct_start 0.05\
      --revin 1\
      --subtract_last 0\
      --affine 1\
      --early_inner_tcn_layers 1\
      --early_outer_tcn_layers 2\
      --inner_tcn_layers 0\
      --outer_tcn_layers 0\
      --final_div_factor 15000 \
      --inner_repatching False \
      --outer_repatching True \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done
done
    #   --shuffle_method 'HalfEx' \