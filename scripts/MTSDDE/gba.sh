if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TSmodel

root_path_name=./dataset/LargeST/GBA/
data_path_name=2017,2018,2019
model_id_name=GBA
data_name=LargeST

seq_len=720
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 2352 \
    --node_num 2352 \
    --subgragh_size 2 \
    --ode_t 1 \
    --train_epochs 10 \
    --patience 5 \
    --itr 1 --batch_size 32 --learning_rate 0.003
done

