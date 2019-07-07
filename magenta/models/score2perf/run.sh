export PYTHONPATH=.
PROBLEM=score2perf_maestro_language_uncropped_aug

#python tensor2tensor/bin/t2t-trainer \
#  --data_dir=./datagen \
#  --problem=${PROBLEM} \
#  --alsologtostderr

gpu=${gpu:-4}
exp_name=${exp_name:-0705_e3}
keep_checkpoint_max=${keep_checkpoint_max:-3}
save_checkpoints_steps=${save_checkpoints_steps:-5000}

if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  gpu=${#t[@]}
fi

echo "Using #gpu=$gpu..."

DATA_DIR=./datagen
HPARAMS_SET=score2perf_transformer_base
MODEL=transformer
PROBLEM=score2perf_maestro_language_uncropped_aug
TRAIN_DIR=./checkpoints/${exp_name}
mkdir -p $TRAIN_DIR
#TRAIN_DIR=./checkpoints/0705_e4_normal

HPARAMS=\
"label_smoothing=0.0,"\
"max_length=0,"\
"max_target_seq_length=4096,"\
"self_attention_type=dot_product_relative_v2,"\
"max_relative_position=1024"

#HPARAMS=\
#"label_smoothing=0.0,"\
#"max_length=0,"\
#"max_target_seq_length=4096"

python tensor2tensor/bin/t2t-trainer \
  --data_dir="${DATA_DIR}" \
  --t2t_usr_dir="magenta/models/score2perf" \
  --hparams=${HPARAMS} \
  --hparams_set=${HPARAMS_SET} \
  --model=${MODEL} \
  --eval_steps=100 \
  --keep_checkpoint_max=$keep_checkpoint_max \
  --local_eval_frequency=5000 \
  --worker_gpu=$gpu \
  --output_dir=${TRAIN_DIR} \
  --problem=${PROBLEM} \
  --iterations_per_loop=$save_checkpoints_steps\
  --train_steps=1000000 2>&1 | tee -a $TRAIN_DIR/log.txt
