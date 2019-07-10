export PYTHONPATH=.
problem=${problem:-score2perf_maestro_language_uncropped_aug}

data_dir=./data/maestro
hparams_set=${hparams_set:-score2perf_transformer_base}
model=transformer
train_dir=./checkpoints/${exp_name}
gpu=${gpu:-4}
exp_name=${exp_name:-0705_e3}
keep_checkpoint_max=${keep_checkpoint_max:-3}
save_checkpoints_steps=${save_checkpoints_steps:-5000}

if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  gpu=${#t[@]}
fi
echo "Using #gpu=$gpu..."
mkdir -p $train_dir

hparams=${hparams:-}

python tensor2tensor/bin/t2t-trainer \
  --data_dir="${data_dir}" \
  --t2t_usr_dir="magenta/models/score2perf" \
  --hparams=${hparams} \
  --hparams_set=${hparams_set} \
  --model=${model} \
  --eval_steps=100 \
  --keep_checkpoint_max=$keep_checkpoint_max \
  --local_eval_frequency=5000 \
  --worker_gpu=$gpu \
  --output_dir=${train_dir} \
  --problem=${problem} \
  --iterations_per_loop=$save_checkpoints_steps\
  --train_steps=1000000 2>&1 | tee -a $train_dir/log.txt
