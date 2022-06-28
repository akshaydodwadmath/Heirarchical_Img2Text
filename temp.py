# External imports
import json
import random
import torch
import os
import argparse
from tqdm import tqdm
from torch.autograd import Variable
from karel.world import World
from itertools import chain
from nps.train import train_seq2seq_model, add_train_cli_args
from karel.world import World

IMG_FEAT = 5184
IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))

actions = [
    'move',
    'turnLeft',
    'turnRight',
    'pickMarker',
    'putMarker',
]

commands = ['REPEAT',
            'WHILE',
            'IF',
            'IFELSE',
            'ELSE',
            ]

def grid_desc_to_tensor(grid_desc):
    grid = torch.Tensor(IMG_FEAT).fill_(0)
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)
    return grid
    
def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]
    
def load_input_file(path_to_dataset, path_to_vocab):
    '''
    path_to_dataset: File containing the data
    path_to_vocab: File containing the vocabulary
    '''
    tgt_tkn2idx = {
        '<pad>': 0,
    }
    next_id = 1
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
             "tkn2idx": tgt_tkn2idx}

    path_to_ds_cache = path_to_dataset.replace('.json', '.txt')
    #with open(path_to_ds_cache, 'r+') as f:
    #    f.truncate(4)
    #if os.path.exists(path_to_ds_cache):
    #    dataset = torch.load(path_to_ds_cache)
    #    print("Entered")
    #else:
    text = ""
    with open(path_to_dataset, 'r') as dataset_file:
        srcs = []
        tgts = []
        current_ios = []
       
        for sample_str in tqdm(dataset_file.readlines()):
            sample_data = json.loads(sample_str)

            # Get the target program
            tgt_program_tkn = sample_data['program_tokens']
            # print('tgt_program_tkn', tgt_program_tkn)
            # print('tgt_program_tkn 3', tgt_program_tkn[-3])
            # print(tgt_program_tkn[3] )
            # print(tgt_program_tkn[4] in actions)
            # print('length tgt_program_tkn', len(tgt_program_tkn))
            
            tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
           
            current_ios = []
            
            rules = [(len(tgt_program_tkn) < 18), (tgt_program_tkn[3] in actions), \
            (tgt_program_tkn[4] in actions) , (tgt_program_tkn[5] in commands), \
            (tgt_program_tkn[-2] in actions), (tgt_program_tkn[-3] in actions)]
            
            if  all(rules):
                for example in sample_data['examples']:
                    inp_grid_coord = []
                    inp_grid_val = []
                    inp_grid_str = example['inpgrid_tensor']
                    for coord_str in inp_grid_str.split():
                        idx, val = coord_str.split(':')
                        inp_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    inp_grid = torch.ShortTensor(inp_grid_coord)
                   

                   #print(inp_grid)
                    #temp_grid = grid_desc_to_tensor(inp_grid)
                    #print(temp_grid)
                    #sample_inp_worlds = []
                    #sample_test_inp_worlds.append(World.fromPytorchTensor(temp_grid))

                    out_grid_coord = []
                    out_grid_val = []
                    out_grid_str = example['outgrid_tensor']
                    for coord_str in out_grid_str.split():
                        idx, val = coord_str.split(':')
                        out_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    out_grid = torch.ShortTensor(out_grid_coord)

                 #   current_ios.append((inp_grid, out_grid))

                srcs.append(current_ios)
                tgts.append(tgt_program_tkn)
               # text += tgt_program_tkn  + "\n"
                with open(path_to_ds_cache, 'a') as f:
                    f.write((' '.join(tgt_program_tkn)))
                    f.write('\n')

    dataset = {"sources": srcs,
               "targets": tgts}
    #torch.save(dataset, path_to_ds_cache)
 

    return dataset, vocab
    
 # Load-up the dataset
 

parser = argparse.ArgumentParser(
    description='Train a Seq2Seq model on type prediction.')
add_train_cli_args(parser)
args = parser.parse_args()
dataset, vocab = load_input_file(args.train_file, args.vocab)