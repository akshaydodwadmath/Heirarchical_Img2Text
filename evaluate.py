from __future__ import division
# External imports
import json
import os
import torch

from torch.autograd import Variable
from tqdm import tqdm

from dataloader import load_input_file_orig, get_minibatch, shuffle_dataset
from karel.consistency import Simulator


def add_eval_args(parser):
    parser.add_argument('--eval_nb_ios', type=int,
                        default=5)
    parser.add_argument('--use_grammar', action="store_true")
    parser.add_argument("--val_nb_samples", type=int,
                        default=0,
                        help="How many samples to use to compute the accuracy."
                        "Default: %(default)s, for all the dataset")
    
def add_beam_size_arg(parser):
    parser.add_argument("--eval_batch_size", type=int,
                        default=1)
    parser.add_argument("--beam_size", type=int,
                        default=10,
                        help="Size of the beam search. Default %(default)s")
    parser.add_argument("--top_k", type=int,
                        default=5,
                        help="How many candidates to return. Default %(default)s")

def add_common_arg(parser):
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use the GPU to run the model")
    parser.add_argument("--intermediate", action="store_true",
                    help="Store Intermediate Grid States for RM")
    parser.add_argument("--log_frequency", type=int,
                        default=100,
                        help="How many minibatch to do before logging"
                        "Default: %(default)s.")
def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]   
    
    
tgt_tkn2idx = {
        '<pad>': 0,
    }
    
def evaluate_model(model_weights,
                   vocabulary_path,
                   dataset_path,
                   nb_ios,
                   nb_samples,
                   use_grammar,
                   output_path,
                   beam_size,
                   top_k,
                   batch_size,
                   use_cuda,
                   intermediate,
                   dump_programs):
                   
    next_id = 1
    path_to_vocab = "data/new_vocab.vocab"
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
             "tkn2idx": tgt_tkn2idx}

    all_semantic_output_path = []
    
    res_dir = os.path.dirname(output_path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for k in range(top_k):
        
        new_semantic_term = "semantic_top%d.txt" % (k+1)
        
        new_semantic_file_name = output_path + new_semantic_term
        

        all_semantic_output_path.append(new_semantic_file_name)
    program_dump_path = os.path.join(res_dir, "generated")
        
    # Load the vocabulary of the trained model
    dataset, vocab = load_input_file_orig(dataset_path, vocabulary_path)
    print("length", len(dataset["targets"]))
    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    simulator = Simulator(vocab["idx2tkn"])
    
    # # Load the model
    # if not use_cuda:
        # # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/8
        # # Is it failing?
        # model = torch.load(model_weights, map_location=lambda storage, loc: storage)
    # else:
        # model = torch.load(model_weights)
        # model.cuda()
        
    # # And put it into evaluation mode
    # model.eval()
    
    # if use_grammar:
        # syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
        # model.set_syntax_checker(syntax_checker)
        
        
    nb_semantic_correct = [0 for _ in range(top_k)]
    nb_generalize_correct = [0 for _ in range(top_k)]
    total_nb = 0
    
    
#    dataset = shuffle_dataset(dataset, batch_size, randomize=False)

    pred_list =[]
        #1    
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'turnRight', 'IFELSE', 'c(', 'noMarkersPresent', 'c)', 'i(', 'putMarker', 'i)', 'ELSE', 'e(', 'pickMarker', 'e)', 'move', 'IF', 'c(', 'markersPresent', 'c)', 'i(', 'pickMarker', 'pickMarker', 'move', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    
        #2  
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'move', 'i)', 'putMarker', 'IF', 'c(', "not", "c(", "rightIsClear", "c)", 'c)', 'i(', 'putMarker', 'i)', 'move', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)    

    #3
    tgt_program_tkn = ['DEF', 'run', 'm(', 'putMarker', 'move', 'putMarker', 'move', 'putMarker', 'move', 'putMarker', 'move', 'putMarker', 'IFELSE', 'c(', "not", "c(", 'rightIsClear', "c)", 'c)', 'i(', 'turnRight', 'turnRight', 'turnRight', 'turnRight', 'turnRight', 'i)', 'ELSE', 'e(', 'move', 'e)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #4
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'turnRight', 'turnRight', 'turnRight', 'turnRight', 'turnRight', 'putMarker', 'turnRight', 'pickMarker', 'pickMarker', 'pickMarker', 'pickMarker', 'WHILE', 'c(', "not", "c(", "frontIsClear", "c)", 'c)', 'w(', 'turnRight', 'move', 'w)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #5
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'pickMarker', 'IFELSE', 'c(', "not", "c(", 'rightIsClear', "c)", 'c)', 'i(', 'turnLeft', 'turnLeft', 'turnLeft', 'turnLeft', 'i)', 'ELSE', 'e(', 'turnRight', 'e)', 'move', 'pickMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #6
    tgt_program_tkn = ['DEF', 'run', 'm(',  'WHILE', 'c(', 'frontIsClear', 'c)', 'w(', 'move', 'putMarker',  'w)', 'turnRight', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #7
    tgt_program_tkn = ['DEF', 'run', 'm(', 'putMarker', 'move', 'turnLeft', 'IF', 'c(', 'markersPresent', 'c)', 'i(', 'putMarker', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #8
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'turnLeft', 'move', 'pickMarker', 'IF', 'c(', 'noMarkersPresent', 'c)', 'i(', 'turnLeft', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #9
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'pickMarker', 'WHILE', 'c(', 'noMarkersPresent', 'c)', 'w(', 'move', 'w)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #10
    tgt_program_tkn = ['DEF', 'run', 'm(', 'putMarker', 'WHILE', 'c(', 'leftIsClear', 'c)', 'w(', 'move', 'w)', 'putMarker', 'move', 'putMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #11
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'move', 'i)', 'putMarker', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'move', 'putMarker', 'putMarker', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)

    #12
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'move', 'IF', 'c(', 'leftIsClear', 'c)', 'i(', 'IF', 'c(', 'frontIsClear', 'c)', 'i(', 'turnRight', 'move', 'i)', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #13
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'turnLeft', 'move', 'pickMarker', 'move', 'IF', 'c(', "not", "c(", 'rightIsClear', 'c)', 'c)', 'i(', 'pickMarker', 'i)', 'move', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #14
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'putMarker', 'IF', 'c(', 'leftIsClear', 'c)', 'i(', 'move', 'turnLeft', 'putMarker', 'turnLeft', 'move', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #15
    tgt_program_tkn = ['DEF', 'run', 'm(', 'move', 'WHILE', 'c(', 'frontIsClear', 'c)', 'w(', 'move', 'w)', 'pickMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #16
    tgt_program_tkn = ['DEF', 'run', 'm(', 'turnLeft', 'putMarker', 'move', 'putMarker', 'move', 'putMarker', 'move', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'turnLeft', 'putMarker', 'move', 'putMarker', 'move', 'putMarker', 'move', 'i)', 'turnRight', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #17
    tgt_program_tkn = ['DEF', 'run', 'm(', 'putMarker', 'move', 'turnRight', 'putMarker', 'move', 'turnRight', 'putMarker', 'move', 'turnRight', 'putMarker', 'move', 'turnRight', 'putMarker', 'move', 'turnRight', 'move', 'move', 'turnRight', 'move', 'turnRight', 'IF', 'c(', 'noMarkersPresent', 'c)', 'i(', 'move', 'i)', 'putMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #18
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'move', 'putMarker', 'putMarker', 'IF', 'c(', "not", "c(", "leftIsClear", "c)", 'c)', 'i(', 'pickMarker', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #19
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'move', 'IF', 'c(',  "not", "c(", "frontIsClear", "c)", 'c)', 'i(', 'turnRight', 'move', 'i)', 'pickMarker', 'move', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'turnLeft', 'i)', 'putMarker', 'turnLeft', 'move', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #20
    tgt_program_tkn = ['DEF', 'run', 'm(', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'pickMarker', 'i)', 'putMarker', 'putMarker', 'move', 'turnRight', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #21
    tgt_program_tkn = ['DEF', 'run', 'm(', 'turnRight', 'move', 'putMarker', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'turnRight', 'i)', 'move', 'putMarker', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'turnRight', 'i)', 'move', 'putMarker', 'IF', 'c(', 'rightIsClear', 'c)', 'i(', 'turnRight', 'i)', 'move', 'putMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #22
    tgt_program_tkn = ['DEF', 'run', 'm(', 'turnLeft', 'move', 'pickMarker', 'IF', 'c(', 'leftIsClear', 'c)', 'i(', 'IF', 'c(', "not", "c(", "frontIsClear", "c)", 'c)', 'i(', 'turnRight', 'pickMarker', 'i)', 'i)', 'move', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #23
    tgt_program_tkn = ['DEF', 'run', 'm(', 'IFELSE', 'c(', 'rightIsClear', 'c)', 'i(', 'turnLeft', 'i)', 'ELSE', 'e(', 'turnRight', 'e)', 'putMarker', 'putMarker', 'turnLeft', 'move', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #24
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'move', 'putMarker', 'IF', 'c(', 'leftIsClear', 'c)', 'i(', 'pickMarker', 'i)', 'putMarker', 'IF', 'c(', "not", "c(", "leftIsClear", "c)", 'c)', 'i(', 'move', 'putMarker', 'i)', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #25
    tgt_program_tkn = ['DEF', 'run', 'm(', 'turnLeft', 'move', 'move', 'IFELSE', 'c(', 'rightIsClear', 'c)', 'i(', 'turnRight', 'i)', 'ELSE', 'e(', 'turnLeft', 'e)', 'putMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
    #26
    tgt_program_tkn = ['DEF', 'run', 'm(', 'pickMarker', 'pickMarker', 'turnLeft', 'move', 'move', 'WHILE', 'c(', 'leftIsClear', 'c)', 'w(', 'turnLeft', 'w)', 'putMarker', 'm)']
    tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    pred_list.append(tgt_program_idces)
        

   # # tgt_program_tkn = ["DEF", "run", "m(", "move", "turnRight", "IF", "c(", "noMarkersPresent", "c)", "i(", "putMarker", "move", "pickMarker", "i)",  "pickMarker",  "move", "m)"]
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "turnRight", "IFELSE", "c(", "noMarkersPresent", "c)", "i(", "putMarker", "i)", "ELSE", "e(", "pickMarker", "e)", "move",    "IF", "c(", "markersPresent", "c)", "i(", "pickMarker",  "pickMarker",  "move", "i)", "m)"]
   # # tgt_program_tkn = ["DEF", "run", "m(", "move", "turnRight", "IF", "c(", "noMarkersPresent", "c)", "i(", "putMarker", "move",  "i)",  "pickMarker",  "IF", "c(", "markersPresent", "c)", "i(",  "pickMarker", "i)", "move", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #3 
    # tgt_program_tkn = ["DEF", "run", "m(", "putMarker", "turnLeft", "move", "putMarker", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #4  
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "REPEAT", "R=3", "r(", "putMarker", "r)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #5 
    # tgt_program_tkn = ["DEF", "run", "m(", "pickMarker", "IF", "c(", "rightIsClear", "c)", "i(", "move", "i)",  "putMarker", "IF", "c(", "not", "c(", "rightIsClear", "c)", "c)", "i(", "putMarker", "i)", "move", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #6  
    # tgt_program_tkn = ["DEF", "run", "m(", "IF", "c(", "noMarkersPresent", "c)", "i(", "turnLeft", "turnLeft","turnLeft","turnLeft","i)", "move", "IF", "c(", "markersPresent", "c)", "i(", "pickMarker", "pickMarker","move","move","i)", "putMarker", "turnLeft",  "turnRight","IF", "c(", "rightIsClear", "c)", "i(", "turnLeft", "i)" , "move", "turnRight", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #7
    # tgt_program_tkn = ["DEF", "run", "m(", "pickMarker", "move", "turnLeft", "turnLeft","m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #8 
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "REPEAT", "R=3", "r(", "IF", "c(", "rightIsClear", "c)", "i(", "IF", "c(", "noMarkersPresent", "c)", "i(", "move", "i)","i)", "r)",  "turnLeft", "IF", "c(", "rightIsClear", "c)", "i(", "IF", "c(", "markersPresent", "c)", "i(", "turnLeft","i)","i)", "pickMarker", "REPEAT", "R=3", "r(",  "IF", "c(", "not", "c(", "rightIsClear", "c)", "c)",  "i(","IF", "c(", "markersPresent", "c)", "i(", "pickMarker", "i)" ,"i)" ,"r)" ,"m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #9  
    # tgt_program_tkn = ["DEF", "run", "m(",  "REPEAT", "R=4", "r(", "putMarker", "move", "r)", "putMarker", "IFELSE", "c(", "not", "c(", "rightIsClear", "c)", "c)", "i(", "turnRight", "turnRight", "turnRight", "turnRight", "turnRight", "i)", "ELSE", "e(", "move", "e)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #10  
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "putMarker",  "putMarker", "turnRight", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
    
    
        # #11    
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "move", "putMarker", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #12  
    # tgt_program_tkn = ["DEF", "run", "m(", "move",  "pickMarker",  "pickMarker", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #13 
    # tgt_program_tkn = ["DEF", "run", "m(", "pickMarker", "move", "putMarker", "move", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #14  
    # tgt_program_tkn = ["DEF", "run", "m(", "pickMarker", "REPEAT", "R=5", "r(", "turnRight", "r)", "putMarker",  "turnRight", "REPEAT", "R=4", "r(", "pickMarker", "r)","turnRight", "move", "WHILE", "c(", "not", "c(", "frontIsClear", "c)", "c)", "w(", "turnRight", "move", "w)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #15 
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "pickMarker", "IFELSE", "c(", "rightIsClear", "c)", "i(", "turnRight", "i)",   "ELSE", "e(",  "turnLeft", "turnLeft", "turnLeft", "turnLeft",  "e)",   "move", "pickMarker", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #16  
    # tgt_program_tkn = ["DEF", "run", "m(", "REPEAT", "R=8", "r(", "turnLeft", "r)", "move","pickMarker", "move", "turnRight" , "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #17
    # tgt_program_tkn = ["DEF", "run", "m(", "turnLeft", "move", "REPEAT",  "R=5", "r(", "putMarker", "r)","WHILE", "c(",  "frontIsClear", "c)",  "w(", "move", "REPEAT", "R=5", "r(", "putMarker", "r)", "w)",  "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #18 
    # tgt_program_tkn = ["DEF", "run", "m(", "REPEAT", "R=4", "r(",  "putMarker", "turnRight", "move", "r)", "move", "turnRight", "move",  "putMarker",  "REPEAT", "R=2", "r(",  "turnLeft", "move", "putMarker",  "turnLeft",  "r)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #19  
    # tgt_program_tkn = ["DEF", "run", "m(",  "WHILE", "c(", "noMarkersPresent", "c)",  "w(", "putMarker", "move", "w)",   "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #20  
    # tgt_program_tkn = ["DEF", "run", "m(", "move", "pickMarker", "turnRight", "move", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
    
    
     # #21    
    # tgt_program_tkn = ["DEF", "run", "m(", "turnRight", "WHILE", "c(", "frontIsClear", "c)", "w(", "move", "putMarker", "w)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #22  
    # tgt_program_tkn = ["DEF", "run", "m(", "move",  "pickMarker",  "move", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #23 
    # tgt_program_tkn = ["DEF", "run", "m(", "REPEAT", "R=3", "r(",  "move", "putMarker", "r)", "WHILE", "c(", "frontIsClear", "c)", "w(", "move", "putMarker", "w)", "turnRight", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
   # #24  
    # tgt_program_tkn = ["DEF", "run", "m(", "putMarker",  "move", "turnLeft", "IF", "c(", "markersPresent", "c)", "i(", "putMarker", "i)", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
       # #25 
    # tgt_program_tkn = ["DEF", "run", "m(",  "turnRight",  "move", "pickMarker", "m)"]
    # tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
    # pred_list.append(tgt_program_idces)
    
    for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
        

        inp_grids, out_grids, \
        inter_grids_1, inter_grids_2, \
        in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
        inp_worlds, out_worlds, \
        inter_worlds_1, inter_worlds_2, \
        targets, \
        inp_test_worlds, out_test_worlds, \
        inter_test_worlds_1, inter_test_worlds_2, \
            target_subprog1, target_subprog2, \
            target_subprog3, \
            input_subprog1,input_subprog2    = get_minibatch(dataset, sp_idx, batch_size,
                                                         tgt_start, tgt_end, tgt_pad,
                                                         nb_ios, simulator,  intermediate, shuffle=False, volatile_vars=True)
    
        # #TODO: WHY?
        # max_len = out_tgt_seq.size(1) + 10
       # # max_len = 6
        # if use_cuda:
            # inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
            # in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
         
        # #TODO: Understand code for dumping programs 
        # if dump_programs:
            # import numpy as np
            # decoder_logit, syntax_logit = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
            # if syntax_logit is not None and model.decoder.learned_syntax_checker is not None:
                # syntax_logit = syntax_logit.cpu().data.numpy()
                # for n in range(in_tgt_seq.size(0)):
                    # decoded_dump_dir = os.path.join(program_dump_path, str(n + sp_idx))
                    # if not os.path.exists(decoded_dump_dir):
                        # os.makedirs(decoded_dump_dir)
                    # seq = in_tgt_seq.cpu().data.numpy()[n].tolist()
                    # seq_len = seq.index(0) if 0 in seq else len(seq)
                    # file_name = str(n) + "_learned_syntax"
                    # norm_logit = syntax_logit[n,:seq_len]
                    # norm_logit = np.log(-norm_logit)
                    # norm_logit = 1 / (1 + np.exp(-norm_logit))
                    # np.save(os.path.join(decoded_dump_dir, file_name), norm_logit)
                    # ini_state = syntax_checker.get_initial_checker_state()
                    # file_name = str(n) + "_manual_syntax"
                    # mask = syntax_checker.get_sequence_mask(ini_state, seq).squeeze().cpu().numpy()[:seq_len]
                    # np.save(os.path.join(decoded_dump_dir, file_name), mask)
                    # file_name = str(n) + "_diff"
                    # diff = mask.astype(float) - norm_logit
                    # diff = (diff + 1) / 2 # remap to [0,1]
                    # np.save(os.path.join(decoded_dump_dir, file_name), diff)
        # batch_list_inputs = [[tgt_start]]*len(targets)
        # decoded = model.beam_sample(inp_grids, out_grids,
                                    # batch_list_inputs, tgt_end, max_len,
                                    # beam_size, top_k)
        
        for batch_idx, (
                        sp_input_worlds, sp_output_worlds,
                        sp_test_input_worlds, sp_test_output_worlds) in \
            enumerate(zip(
                          inp_worlds, out_worlds,
                          inp_test_worlds, out_test_worlds,)):
        #    print("Entereddddddddd")

            total_nb += 1 #should be batch size * number of IOs
           ## target = target.cpu().data.squeeze().numpy().tolist()
           ## target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]
                
                
                # if dump_programs:
                    # decoded_dump_dir = os.path.join(program_dump_path, str(batch_idx + sp_idx))
                    # if not os.path.exists(decoded_dump_dir):
                        # os.makedirs(decoded_dump_dir)
                    # write_program(os.path.join(decoded_dump_dir, "target"), target, vocab["idx2tkn"])
                    # for rank, dec in enumerate(sp_decoded):
                        # pred = dec[1]
                        # ll = dec[0]
                        # file_name = str(rank)+ " - " + str(ll)
                        # write_program(os.path.join(decoded_dump_dir, file_name), pred, vocab["idx2tkn"])

          
          #  print("idces", tgt_program_idces)
          #  print("targets", targets)
    
    
    
            # Semantic matches
            #for rank, dec in enumerate(sp_decoded):
            #for i in range(0,10):
                
            pred = pred_list[sp_idx]
           
            parse_success, cand_prog = simulator.get_prog_ast(pred)
            if (not parse_success):
                print("parsing failed")
                continue
            semantically_correct = True
            for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                res_emu = simulator.run_prog(cand_prog, input_world)
                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                    # This prediction is semantically incorrect.
                    print("semantically not correct")
                    semantically_correct = False
              #      break
            if semantically_correct:
                print("semantically correct")
                # Score for all the following ranks
                #for top_idx in range(rank, top_k):
                nb_semantic_correct[0] += 1
             #   break
                
                
            # Generalization
         #   for rank, dec in enumerate(sp_decoded):
            pred = pred_list[sp_idx]
            print("IDX, pred", sp_idx, pred_list[sp_idx])
            parse_success, cand_prog = simulator.get_prog_ast(pred)
            if (not parse_success):
                print("gparsing failed")
                continue
            generalizes = True
            for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                res_emu = simulator.run_prog(cand_prog, input_world)
                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                   
                    # This prediction is semantically incorrect.
                    generalizes = False
                    break
            for (input_world, output_world) in zip(sp_test_input_worlds, sp_test_output_worlds):
                res_emu = simulator.run_prog(cand_prog, input_world)
                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                    # This prediction is semantically incorrect.
                    print("generalizes not correct")
                    generalizes = False
                    break
            if generalizes:
                # Score for all the following ranks
               # for top_idx in range(rank, top_k):
                nb_generalize_correct[0] += 1
                break
                
    # for k in range(top_k):
        # with open(str(all_semantic_output_path[k]), "w") as sem_res_file:
            # sem_res_file.write(str(100*nb_semantic_correct[k]/total_nb))
            
    semantic_at_one = 100*nb_semantic_correct[0]/total_nb
    generatlize_at_one = 100*nb_generalize_correct[0]/total_nb
    print("semantic_at_one", semantic_at_one)
    print("generatlize_at_one", generatlize_at_one)
    return semantic_at_one

def write_program(path, tkn_idxs, vocab):
    program_tkns = [vocab[tkn_idx] for tkn_idx in tkn_idxs]

    indent = 0
    is_new_line = False
    with open(path, "w") as target_file:
        for tkn in program_tkns:
            if tkn in ["m(", "w(", "i(", "e(", "r("]:
                indent += 4
                target_file.write("\n"+" "*indent)
                target_file.write(tkn + " ")
                is_new_line = False
            elif tkn in ["m)", "w)", "i)", "e)", "r)"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                indent -= 4
                target_file.write(tkn)
                if indent < 0:
                    indent = 0
                is_new_line = True
            elif tkn in ["REPEAT"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
            else:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
        target_file.write("\n")
