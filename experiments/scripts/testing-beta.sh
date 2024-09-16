save=$1
model=$2
lang=$3
beta=$4
batch=20
gamma_tr=0.4
gamma_gu=0.0
n=256
buffer=100
seed=1

export PYTHONPATH="."

python experiments/c4-experiment.py --save $save/mod-beta-$beta.p --n $n --method gumbel_mod --batch_size $batch --m 50 --beta $beta --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
