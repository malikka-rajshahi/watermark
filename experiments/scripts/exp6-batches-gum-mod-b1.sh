#!/bin/sh
save=$1
model=$2
lang=$3
m=$4
batch=20
gamma_tr=0.4
gamma_gu=0.0
n=256
buffer=100
seed=1

export PYTHONPATH="."

#    python experiments/c4-experiment.py --save $save/$lang-run-$run-trst.p --n $n --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
#    python experiments/c4-experiment.py --save $save/$lang-m-$m-gust.p --n $n --method gumbel --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
python experiments/c4-experiment.py --save $save/$lang-m-$m-mod-b1.p --n $n --method gumbel_mod --batch_size $batch --m 15 --rt_translate --model $model --language $lang --beta 0.1 --buffer_tokens $buffer --seed $seed
#    python experiments/c4-experiment.py --save $save/$lang-m-$m-mod-b2.p --n $n --method gumbel_mod --batch_size $batch --m $m --rt_translate --model $model --language $lang --beta 1.0 --buffer_tokens $buffer --seed $seed
#    python experiments/c4-experiment.py --save $save/$lang-m-$m-mod-b3.p --n $n --method gumbel_mod --batch_size $batch --m $m --rt_translate --model $model --language $lang --beta 10.0 --buffer_tokens $buffer --seed $seed
