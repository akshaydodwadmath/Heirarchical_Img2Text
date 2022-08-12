Usage: 

Supervised Learning:
python3 main.py --train_file=data/train.json --val_file=data/val.json --batch_size=128  --nb_epochs=100


RL Training (for GPU):
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforce_finetune --batch_size 16 --nb_rollouts 100 --nb_epochs 100 --use_cuda --intermediate

Evaluation:
python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/train.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/fake_run/Results/TrainSet_ --beam_size 64 --top_k 10 --dump_programs 


