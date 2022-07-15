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
                    nb_rollouts, rm_state):

    # Samples `nb_rollouts` samples from the decoding model.
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [], rm_state)
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

def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad
