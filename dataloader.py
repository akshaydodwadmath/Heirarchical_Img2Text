# External imports
import json
import random
import torch
import os

from tqdm import tqdm
from torch.autograd import Variable
from karel.world import World
from itertools import chain


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

tgt_tkn2idx = {
        '<pad>': 0,
    }
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
            
            rules = [(len(tgt_program_tkn) == 15), (tgt_program_tkn[3] in actions), \
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

                    current_ios.append((inp_grid, out_grid))

                srcs.append(current_ios)
                tgts.append(tgt_program_tkn)
               # text += tgt_program_tkn  + "\n"
                with open(path_to_ds_cache, 'w') as f:
                    f.write((' '.join(tgt_program_tkn)))
                    f.write('\n')

    

    dataset = {"sources": srcs,
               "targets": tgts}
    #torch.save(dataset, path_to_ds_cache)
 

    return dataset, vocab
    
def load_input_file_orig(path_to_dataset, path_to_vocab):
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

    path_to_ds_cache = path_to_dataset.replace('.json', '.thdump')
    if os.path.exists(path_to_ds_cache):
        dataset = torch.load(path_to_ds_cache)
    else:
        with open(path_to_dataset, 'r') as dataset_file:
            srcs = []
            tgts = []
            current_ios = []
            for sample_str in tqdm(dataset_file.readlines()):
                sample_data = json.loads(sample_str)

                # Get the target program
                tgt_program_tkn = sample_data['program_tokens']

                tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
                current_ios = []

                for example in sample_data['examples']:
                    inp_grid_coord = []
                    inp_grid_val = []
                    inp_grid_str = example['inpgrid_tensor']
                    for coord_str in inp_grid_str.split():
                        idx, val = coord_str.split(':')
                        inp_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    inp_grid = torch.ShortTensor(inp_grid_coord)

                    out_grid_coord = []
                    out_grid_val = []
                    out_grid_str = example['outgrid_tensor']
                    for coord_str in out_grid_str.split():
                        idx, val = coord_str.split(':')
                        out_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    out_grid = torch.ShortTensor(out_grid_coord)

                    current_ios.append((inp_grid, out_grid))

                srcs.append(current_ios)
                tgts.append(tgt_program_idces)

        dataset = {"sources": srcs,
                   "targets": tgts}
        torch.save(dataset, path_to_ds_cache)

    return dataset, vocab

#TODO: undestand this
def shuffle_dataset(dataset, batch_size, randomize=True):
    '''
    We are going to group together samples that have a similar length, to speed up training
    batch_size is passed so that we can align the groups
    '''
    pairs = list(zip(dataset["sources"], dataset["targets"]))
    bucket_fun = lambda x: len(x[1]) / 5
    pairs.sort(key=bucket_fun, reverse=True)
    grouped_pairs = [pairs[pos: pos + batch_size]
                     for pos in range(0,len(pairs), batch_size)]
    if randomize:
        to_shuffle = grouped_pairs[:-1]
        random.shuffle(to_shuffle)
        grouped_pairs[:-1] = to_shuffle
    pairs = chain.from_iterable(grouped_pairs)
    in_seqs, out_seqs = zip(*pairs)
    return {
        "sources": in_seqs,
        "targets": out_seqs
    }

def get_minibatch(dataset, sp_idx, batch_size,
                  start_idx, end_idx, pad_idx,
                  nb_ios, simulator, intermediate, shuffle=True, volatile_vars=False):
    """Prepare minibatch."""

    # Prepare the grids
    grid_descriptions = dataset["sources"][sp_idx:sp_idx+batch_size]
    
    # Prepare the target sequences
    targets = dataset["targets"][sp_idx:sp_idx+batch_size]   
    
    inp_grids = []
    out_grids = []
    inter_grids_1 = []
    inter_grids_2 = []
    
    
    inp_worlds= []
    out_worlds= []
    
    
    inter_worlds_1 = []
    inter_worlds_2 = []
    inter_test_worlds_1 = []
    inter_test_worlds_2 = []
    
    input_subprog1 = []
    input_subprog2 = []

    
    target_subprog1 = []
    target_subprog2 = []
    target_subprog3 = []
    
    inp_test_worlds = []
    out_test_worlds = []
    
    for sample, sample_target in zip(grid_descriptions,targets):
        if shuffle:
            random.shuffle(sample)
        sample_inp_grids = []
        sample_out_grids = []
        if (intermediate):
            sample_inter_grids_1 = []
            sample_inter_grids_2 = []
        
        sample_inp_worlds = []
        sample_out_worlds = []

        sample_test_inp_worlds = []
        sample_test_out_worlds = []
        for inp_grid_desc, out_grid_desc in sample[:nb_ios]:

            # Do the inp_grid
            inp_grid = grid_desc_to_tensor(inp_grid_desc)
            # Do the out_grid
            out_grid = grid_desc_to_tensor(out_grid_desc)

            sample_inp_grids.append(inp_grid)
            sample_out_grids.append(out_grid)
            #TODO: Understand World Class
            sample_inp_worlds.append(World.fromPytorchTensor(inp_grid))
            sample_out_worlds.append(World.fromPytorchTensor(out_grid))


            
            
            #for sample_world in sample_inter_worlds_1:
                #sample_inter_grids_1.append(World.toPytorchTensor(sample_world, IMG_DIM))
                #sample_inter_grids_2.append(World.toPytorchTensor(sample_inter_worlds_2, IMG_DIM))

        for inp_grid_desc, out_grid_desc in sample[nb_ios:]:
            # Do the inp_grid
            inp_grid = grid_desc_to_tensor(inp_grid_desc)
            # Do the out_grid
            out_grid = grid_desc_to_tensor(out_grid_desc)
      
            sample_test_inp_worlds.append(World.fromPytorchTensor(inp_grid))
            sample_test_out_worlds.append(World.fromPytorchTensor(out_grid))
            
        if (intermediate):
            subprog_1, subprog_2, subprog_3 = get_intermediate_prog(sample_target)
            
            sample_inter_worlds_1, sample_inter_worlds_2, \
                sample_inter_grids_1, sample_inter_grids_2 =  get_intermediate_grids(sample_inp_worlds, sample_out_worlds, subprog_1,subprog_2,subprog_3, simulator)
            
            sample_test_inter_worlds_1, sample_test_inter_worlds_2, \
                 _, _ =  get_intermediate_grids(sample_test_inp_worlds, sample_test_out_worlds, subprog_1,subprog_2,subprog_3, simulator)


            input_subprog1.append([start_idx] + subprog_1[:-1])
            input_subprog2.append([start_idx] + subprog_1[:-1] + subprog_2[3:-1])
            
            target_subprog1.append(subprog_1)
            target_subprog2.append(subprog_1[:-1] + subprog_2[3:])

        
        sample_inp_grids = torch.stack(sample_inp_grids, 0)
        sample_out_grids = torch.stack(sample_out_grids, 0)
        inp_grids.append(sample_inp_grids)
        out_grids.append(sample_out_grids)

        if (intermediate):
            sample_inter_grids_1 = torch.stack(sample_inter_grids_1, 0)
            sample_inter_grids_2 = torch.stack(sample_inter_grids_2, 0)
            inter_grids_1.append(sample_inter_grids_1)
            inter_grids_2.append(sample_inter_grids_2)
        
        inp_worlds.append(sample_inp_worlds)
        out_worlds.append(sample_out_worlds)
        
        
        if (intermediate):
            inter_worlds_1.append(sample_inter_worlds_1)
            inter_worlds_2.append(sample_inter_worlds_2)
            inter_test_worlds_1.append(sample_test_inter_worlds_1)
            inter_test_worlds_2.append(sample_test_inter_worlds_2)
        
        
        inp_test_worlds.append(sample_test_inp_worlds)
        out_test_worlds.append(sample_test_out_worlds)
    inp_grids = Variable(torch.stack(inp_grids, 0), volatile=volatile_vars)
    out_grids = Variable(torch.stack(out_grids, 0), volatile=volatile_vars)
    if (intermediate):
        inter_grids_1 = Variable(torch.stack(inter_grids_1, 0), volatile=volatile_vars)
        inter_grids_2 = Variable(torch.stack(inter_grids_2, 0), volatile=volatile_vars)
   
    lines = [
        [start_idx] + line for line in targets
    ]
    lens = [len(line) for line in lines]
    max_len = max(lens)

    # Drop the last element, it should be the <end> symbol for all of them
    # padding for all of them
    input_lines = [
        line[:max_len-1] + [pad_idx] * (max_len - len(line[:max_len-1])-1) for line in lines
    ]

    # Drop the first element, should always be the <start> symbol. This makes
    # everything shifted by one compared to the input_lines
    output_lines = [
        line[1:] + [pad_idx] * (max_len - len(line)) for line in lines
    ]

    in_tgt_seq = Variable(torch.LongTensor(input_lines), volatile=volatile_vars)
    #print('in_tgt_seq', in_tgt_seq)
    out_tgt_seq = Variable(torch.LongTensor(output_lines), volatile=volatile_vars)
 
    return inp_grids, out_grids, inter_grids_1, inter_grids_2, in_tgt_seq, input_lines, out_tgt_seq, \
        inp_worlds, out_worlds, inter_worlds_1, inter_worlds_2, targets, inp_test_worlds, out_test_worlds, inter_test_worlds_1, inter_test_worlds_2, \
            target_subprog1, target_subprog2, target_subprog3, input_subprog1, input_subprog2
    
####Idea of passing almost full to no inputs
#def get_intermediate_prog_2(line , i):
    #return line[:-i]

def get_intermediate_prog(line):
    token_beg = ["DEF", "run", "m("] 
    token_end = ["m)"]
    subprog_1 = []
    subprog_2 = []
    subprog_3 = []
    
    token_beg = translate(token_beg, tgt_tkn2idx)
    token_end = translate(token_end, tgt_tkn2idx)
    subprog_1 = line[:5] + token_end 
    subprog_2 = token_beg + line[5:-3] + token_end 
    subprog_3 = token_beg + line[-3:]   

    return subprog_1, subprog_2, subprog_3

def get_intermediate_grids(inp_worlds, out_worlds, subprog_1, subprog_2, subprog_3, simulator):
    inter_1 = []
    inter_2 = []
    inter_grid_1 = []
    inter_grid_2 = []
    ## Make sure that the reference program works for the IO given
    #parse_success, ref_prog = self.simulator.get_prog_ast(subprog_1)
    #assert(parse_success)
    correct_reference = True
    for inp_world, out_world in zip(inp_worlds, out_worlds):
        
        #Run Sub Program 1
        parse_success, cand_prog = simulator.get_prog_ast(subprog_1)
        if (not parse_success):
                raise Exception("Parsing failed")
        res_emu = simulator.run_prog(cand_prog, inp_world)
        correct_reference = correct_reference and (res_emu.status == 'OK')
        correct_reference = correct_reference and (not res_emu.crashed)
        if(correct_reference):
            temp_1 = res_emu.outgrid
        else:
            raise Exception("Parsing failed")
            
        
        #Run Sub Program 2
        parse_success, cand_prog = simulator.get_prog_ast(subprog_2)
        if (not parse_success):
                raise Exception("Parsing failed")
        res_emu = simulator.run_prog(cand_prog, temp_1)
        correct_reference = correct_reference and (res_emu.status == 'OK')
        correct_reference = correct_reference and (not res_emu.crashed)
        if(correct_reference):
            temp_2 = res_emu.outgrid
        else:
             raise Exception("Parsing failed")
        
        
        #Run Sub Program 3
        parse_success, cand_prog = simulator.get_prog_ast(subprog_3)
        if (not parse_success):
                raise Exception("Parsing failed")
        res_emu = simulator.run_prog(cand_prog, temp_2)
        correct_reference = correct_reference and (res_emu.status == 'OK')
        correct_reference = correct_reference and (not res_emu.crashed)
        correct_reference = correct_reference and (out_world == res_emu.outgrid)
        if(not correct_reference):
            raise Exception("Parsing failed")
            
        inter_1.append(temp_1)
        inter_2.append(temp_2)
        inter_grid_1.append(World.toPytorchTensor(temp_1, IMG_DIM))
        inter_grid_2.append(World.toPytorchTensor(temp_2, IMG_DIM))
        
    return inter_1,inter_2, inter_grid_1, inter_grid_2
