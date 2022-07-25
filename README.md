Usage: python3 main.py --train_file=data/train.json --val_file=data/val.json --batch_size=128  --nb_epochs=100

Usage for generating selected data: python3 main.py --train_file=data/train_data.json --val_file=data/val_data.json --batch_size=16  --nb_epochs=5 --save_to_txt

Evaluation(Supervised):
python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/test.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/fake_run/Results/TestSet_ --beam_size 64 --top_k 10 

python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/val.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/fake_run/Results/TestSet_ --beam_size 64 --top_k 10 --dump_programs 

RL Training:
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforce_finetune --batch_size 16  --nb_epochs 5 --nb_rollouts 100

RL Training for GPU:
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforce_finetune --batch_size 16 --nb_rollouts 100 --nb_epochs 5 --use_cuda


Reward Machine Included:

RL Training:
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforce_finetune --batch_size 16  --nb_epochs 5 --nb_rollouts 100 --intermediate

RL Training for GPU:
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforce_finetune --batch_size 16 --nb_rollouts 100 --nb_epochs 5 --use_cuda --intermediate

python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/test.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/fake_run/Results/TestSet_ --beam_size 64 --top_k 10 --intermediate

RL Beam RL Training:
python3 main.py  --signal beam_rl --learning_rate 1e-5 --init_weights exps_150epochs_sup_owndatset_smallermodel/fake_run/Weights/weights_99.model --train_file data/train.json --result_folder exps/reinforcebeamRL_finetune --batch_size 16  --nb_epochs 5  --intermediate --rl_inner_batch 8 --rl_use_ref --rl_beam 64 --use_cuda
