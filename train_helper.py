import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable

def do_supervised_minibatch(model,
                            # Source
                            inp_grids, out_grids,
                            # Target
                            in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                            # Criterion
                            criterion):

    # Get the log probability of each token in the ground truth sequence of tokens.
    decoder_logit, _ = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
    
    nb_predictions = torch.numel(out_tgt_seq.data)
    
    ## out_tgt_seq will be of size batch size x max seq length, non max sequences will be padded
    ## decoder_logit will be of size (out_tgt_seq size, vocab size(52))
    ## cross entropy loss requires unnormalized prediction scores of all tokens, output class

    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # Do the backward pass over the loss
    loss.backward()
    
    print('loss', loss.item())

    # Return the value of the loss over the minibatch for monitoring
    return loss.item()
    
    
def do_rl_minibatch(model,
                    # Source
                    inp_grids, out_grids,
                    # Target
                    envs,
                    # Config
                    tgt_start_idx, tgt_end_idx, max_len,
                    nb_rollouts):

    # Samples `nb_rollouts` samples from the decoding model.
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [])
        #print('roll.dep_reward', roll.dep_reward)

    # Evaluate the performance on the minibatch
    batch_reward = sum(roll.dep_reward for roll in rolls)
    # Get all variables and all gradients from all the rolls
    variables, grad_variables = zip(*batch_rolls_reinforce(rolls))

    # For each of the sampling probability, we know their gradients.
    # See https://arxiv.org/abs/1506.05254 for what we are doing,
    # simply using the probability of the choice made, times the reward of all successors.
    autograd.backward(variables, grad_variables)

    # Return the value of the loss/reward over the minibatch for convergence
    # monitoring.
    return batch_reward

def do_beam_rl(model,
               # source
               inp_grids, out_grids, targets,
               # Target
               envs, reward_comb_fun,
               # Config
               tgt_start_idx, tgt_end_idx, pad_idx,
               max_len, beam_size, rl_inner_batch, rl_use_ref):
    '''
    Rather than doing an actual expected reward,
    evaluate the most likely programs using a beam search (with `beam_sample`)
    If `rl_use_ref` is set, include the reference program in the search.
    Similarly to `do_rl_minibatch_two_steps`, first decode the programs as Volatile,
    then score them.
    '''
    batch_reward = 0
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    vol_inp_grids = Variable(inp_grids.data)
    vol_out_grids = Variable(out_grids.data)
    # Get the programs from the beam search
    decoded = model.beam_sample(vol_inp_grids, vol_out_grids,
                                tgt_start_idx, tgt_end_idx, max_len,
                                beam_size, beam_size)

    # For each element in the batch, get the version of the log proba that can use autograd.
    for start_pos in range(0, len(decoded), rl_inner_batch):
        to_score = decoded[start_pos: start_pos + rl_inner_batch]
        scorers = envs[start_pos: start_pos + rl_inner_batch]
        # Eventually add the reference program
        if rl_use_ref:
            references = [target for target in targets[start_pos: start_pos + rl_inner_batch]]
            for ref, candidates_to_score in zip(references, to_score):
                for _, predded in candidates_to_score:
                    if ref == predded:
                        break
                else:
                    candidates_to_score.append((None, ref)) # Don't know its lpb

        # Build the inputs to be scored
        nb_cand_per_sp = [len(candidates) for candidates in to_score]
        in_tgt_seqs = []
        preds  = [pred for lp, pred in itertools.chain(*to_score)]
        lines = [[tgt_start_idx] + line  for line in preds]
        lens = [len(line) for line in lines]
        ib_max_len = max(lens)

        inp_lines = [
            line[:ib_max_len-1] + [pad_idx] * (ib_max_len - len(line[:ib_max_len-1])-1) for line in lines
        ]
        out_lines = [
            line[1:] + [pad_idx] * (ib_max_len - len(line)) for line in lines
        ]
        in_tgt_seq = Variable(torch.LongTensor(inp_lines))
        out_tgt_seq = Variable(torch.LongTensor(out_lines))
        if use_cuda:
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
        out_care_mask = (out_tgt_seq != pad_idx)

        inner_batch_in_grids = inp_grids.narrow(0, start_pos, len(to_score))
        inner_batch_out_grids = out_grids.narrow(0, start_pos, len(to_score))

        # Get the scores for the programs we decoded.
        seq_lpb_var = model.score_multiple_decs(inner_batch_in_grids,
                                                inner_batch_out_grids,
                                                in_tgt_seq, inp_lines,
                                                out_tgt_seq, nb_cand_per_sp)
        lpb_var = torch.mul(seq_lpb_var, out_care_mask.float()).sum(1)

        # Compute the reward that were obtained by each of the sampled programs
        per_sp_reward = []
        for env, all_decs in zip(scorers, to_score):
            sp_rewards = []
            for (lpb, dec) in all_decs:
                sp_rewards.append(env.step_reward(dec, True))
            per_sp_reward.append(sp_rewards)

        per_sp_lpb = []
        start = 0
        for nb_cand in nb_cand_per_sp:
            per_sp_lpb.append(lpb_var.narrow(0, start, nb_cand))
            start += nb_cand

        # Use the reward combination function to get our loss on the minibatch
        # (See `reinforce.py`, possible choices are RenormExpected and the BagExpected)
        inner_batch_reward = 0
        for pred_lpbs, pred_rewards in zip(per_sp_lpb, per_sp_reward):
            inner_batch_reward += reward_comb_fun(pred_lpbs, pred_rewards)

        # We put a minus sign here because we want to maximize the reward.
        (-inner_batch_reward).backward()

        batch_reward += inner_batch_reward.item()

    return batch_reward

def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad
