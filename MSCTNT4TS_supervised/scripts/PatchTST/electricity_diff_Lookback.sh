if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=336
model_name=PatchTST

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity_diff_Lookback2
data_name=custom

random_seed=2021


for seq_len in 48 96 144 192 336 512 672 720
do 

# for pred_len in 96 192 336 720
for pred_len in 96
# for pred_len in 144 512 672
# for pred_len in 512 672
# for pred_len in 672
# for pred_len in 144
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
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0001 \
      | tee -a logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done