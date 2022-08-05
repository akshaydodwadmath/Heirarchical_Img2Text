# External imports
import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


RMStates = {
    "Full": 0,
    "InterGrid1": 1,
    "InterGrid2": 2,
}
class Rolls(object):

    def __init__(self, action, proba, multiplicity, depth):
        self.successor = {}
        # The action that this node in the tree corresponds to
        self.action = action  # -> what was the sample taken here
        self.proba = proba  # -> Variable containing the log proba of
                            # taking this action
        self.multi_of_this = multiplicity  # -> How many times was this
                                              # prefix (until this point of
                                              # the sequence) seen
        self.depth = depth  # -> How far along are we in the sequence

        # Has no successor to this sample
        self.is_final = True

        # This is the reward that would be obtained by doing this prefix once.
        # This is only to use for bookkeeping.
        self.own_reward = 0
        # The one to use to compute gradients is the following.

        # This contains `self.own_reward * self.multi_of_this` + sum of all the
        # dependents self.dep_reward
        self.dep_reward = 0

    ## Stores the trajectories(program tokens) as well as successor to each trajectory token
    def expand_samples(self, trajectory, end_multiplicity, end_proba):
        '''
        The assumption here is that all but the last steps of the trajectory
        have already been created.
        '''
        assert(len(trajectory) > 0)

        pick = trajectory[0]
        if pick in self.successor:
         #   print('pick', pick)
            self.successor[pick].expand_samples(trajectory[1:],
                                                end_multiplicity,
                                                end_proba)
        else:
            # We add a successor so we are necessarily not final anymore
            self.is_final = False
            # We don't expand the samples by several steps at a time so verify
            # that we are done
            assert(len(trajectory) == 1)
            self.successor[pick] = Rolls(pick, end_proba,
                                         end_multiplicity,
                                         self.depth + 1)
            
        ##TODEBUG
        #print('self.successor', self.successor)

    def yield_var_and_grad(self):
        '''
        Yields 2-tuples:
        -> Proba: Variable correponding to the proba of this last choice
        -> Grad: Gradients for each of those variables
        '''
        for succ in self.successor.values():
            for var, grad in succ.yield_var_and_grad():
                yield var, grad
        yield self.proba, self.reinforce_gradient()

    def assign_rewards(self, reward_assigner, trace, rm_state):
        '''
        Using the `reward_assigner` scorer, go depth first to assign the
        reward at each timestep, and then collect back all the "depending
        rewards"
        '''
        if self.depth == -1:
            # This is the root from which all the samples come from, ignore
            pass
        else:
            # Assign to this step its own reward
            self.own_reward = reward_assigner.step_reward(trace,rm_state,
                                                          self.is_final)

        # Assign their own score to each of the successor
        for next_step, succ in self.successor.items():
            new_trace = trace + [next_step.item()]
            
            ##TODEBUG
            #print('new_trace',new_trace) 
            succ.assign_rewards(reward_assigner, new_trace, rm_state)

        # If this is a final node, there is no successor, so I can already
        # compute the dep-reward.
        if self.is_final:
            self.dep_reward = self.multi_of_this * self.own_reward
        else:
            # On the other hand, all my child nodes have already computed their
            # dep_reward so I can collect them to compute mine
            self.dep_reward = self.multi_of_this * self.own_reward
            for succ in self.successor.values():
                self.dep_reward += succ.dep_reward

    def reinforce_gradient(self):
        '''
        At each decision, compute a reinforce gradient estimate to the
        parameter of the probability that was sampled from.
        '''
        if self.depth == -1:
            return None
        else:
            # We haven't put in a baseline so just ignore this
            baselined_reward = self.dep_reward
            grad_value = baselined_reward / (1e-6 + self.proba.data)

            # We return a negative here because we want to maximize the rewards
            # And the pytorch optimizers try to minimize them, so this put them
            # in agreement
            return -grad_value



class Environment(object):

    def __init__(self, reward_norm, environment_data):
        '''
        reward_norm: float -> Value of the reward for correct answer
        environment_data: anything -> Data/Ground Truth to use for the reward evaluation


        To create different types of reward, subclass it and modify the
        `should_skip_reward` and `reward_value` function.
        '''
        self.reward_norm = reward_norm
        self.environment_data = environment_data

    def step_reward(self, trace, rm_state, is_final):
        '''
        trace: List[int] -> all prediction of the sample to score.
        is_final: bool -> Is the sample finished.
        '''
        if self.should_skip_reward(trace, rm_state, is_final):
            return 0
        else:
            return self.reward_value(trace, rm_state, is_final)

    def should_skip_reward(self, trace, rm_state, is_final):
        raise NotImplementedError

    def reward_value(self, trace, rm_state, is_final):
        raise NotImplementedError

##State: Input and Output input_worlds
##Action: Program
##Reward: +1 or -1

class MultiIOGrid(Environment):
    '''
    This only gives rewards at the end of the prediction.
    +1 if the two programs lead to the same final state.
    -1 if the two programs lead to different outputs
    '''

    def __init__(self, reward_norm,
                 target_program, input_worlds, output_worlds, inter_worlds_1, inter_worlds_2, iterate, simulator):
        '''
        reward_norm: float
        input_grids, output_grids: Reference IO for the synthesis
        '''
        super(MultiIOGrid, self).__init__(reward_norm,
                                        (target_program,
                                         input_worlds,
                                         output_worlds,
                                         inter_worlds_1,
                                         inter_worlds_2,
                                         iterate,
                                         simulator))
        self.target_program = target_program
        
        if(iterate == 0):
            self.input_worlds = input_worlds
            self.output_worlds = inter_worlds_1
        elif(iterate == 1):
            self.input_worlds = input_worlds
            self.output_worlds = inter_worlds_2
        else:
            self.input_worlds = input_worlds
            self.output_worlds = output_worlds
        self.inter_worlds_1 = inter_worlds_1
        self.inter_worlds_2 = inter_worlds_2
        self.simulator = simulator

        # Make sure that the reference program works for the IO given
        parse_success, ref_prog = self.simulator.get_prog_ast(self.target_program)
        assert(parse_success)
        self.correct_reference = True
        self.ref_actions_taken = 1
        for inp_world, out_world in zip(self.input_worlds, self.output_worlds):
            res_emu = self.simulator.run_prog(ref_prog, inp_world)

            self.correct_reference = self.correct_reference and (res_emu.status == 'OK')
            self.correct_reference = self.correct_reference and (not res_emu.crashed)
            self.correct_reference = self.correct_reference and (out_world == res_emu.outgrid)
            self.ref_actions_taken = max(self.ref_actions_taken, len(res_emu.actions))

    def should_skip_reward(self, trace, rm_state,is_final):
        return (not is_final)

    def reward_value(self, trace, rm_state, is_final):
       # print("Trace", trace)
        if (not self.correct_reference):
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0
        rew = 0
        #if(rm_state == 1):
            #trace = trace[:5] + [21]
       # print("rm_state",rm_state)
    
     #   print("prev trace", trace)
        if(rm_state == 0):
            trace = trace + [21]
        elif(rm_state == 1):
            trace = [3,4,20] + trace[5:] + [21]
        else:
            trace = [3,4,20] + trace[12:]
     #   print("trace", trace)
        
        parse_success, cand_prog = self.simulator.get_prog_ast(trace)

        ##TODEBUG
        #inter_trace_1 = trace[:5] + [21]
        #parse_success_notimp, cand_prog_inter_1  = self.simulator.get_prog_ast(inter_trace_1)
        if ((not parse_success)):
            # Program is not syntactically correct
            rew = -self.reward_norm
        else:
            for inp_world, out_world, inter_worlds_1, inter_worlds_2 in zip(self.input_worlds, self.output_worlds, self.inter_worlds_1, self.inter_worlds_2):
               # print("self.input_worlds", len(self.input_worlds))
                #if ( rm_state == 1):
                    #test_world = inter_worlds_1
                if ( rm_state == 0):
                    test_world_1 = inp_world
                    test_world_2 = inter_worlds_1
                elif ( rm_state == 1):
                    test_world_1 = inter_worlds_1
                    test_world_2 = inter_worlds_2
                else:
                    test_world_1 = inter_worlds_2
                    test_world_2 = out_world
                    
                res_emu = self.simulator.run_prog(cand_prog, test_world_1)
                if res_emu.status != 'OK' or res_emu.crashed:
                    # Crashed or failed the simulator
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                elif res_emu.outgrid != test_world_2:
                    # Generated a wrong state
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                else:
                    #res_emu = self.simulator.run_prog(cand_prog, inp_world)
                    #if res_emu.status != 'OK' or res_emu.crashed:
                        ## Crashed or failed the simulator
                        ## Set the reward to negative and stop looking
                        #rew = -self.reward_norm *0.1
                        #break
                    #elif res_emu.outgrid != out_world:
                        ## Generated a wrong state
                        ## Set the reward to negative and stop looking
                        #rew = -self.reward_norm *0.1
                        #break
                    #else:
                    rew = self.reward_norm
  
        return rew
        
        
    def reward_value_rm(self, trace, is_final):
        
        if (not self.correct_reference):
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0
        rew = 0
        parse_success, cand_prog = self.simulator.get_prog_ast(trace)
        ##TODEBUG
        inter_trace_1 = trace[:5] + [21]
        inter_trace_2 = trace[:-3] + [21]
        inter_trace_3 = trace
        
        
        parse_success_notimp, cand_prog_inter_1  = self.simulator.get_prog_ast(inter_trace_1)
        if ((not parse_success) or (not parse_success_notimp)):
            # Program is not syntactically correct
           # rew = self.reward_norm* 0.8
            rew = -self.reward_norm
        else:
            for inp_world, out_world, inter_worlds_1, inter_worlds_2 in zip(self.input_worlds, self.output_worlds, self.inter_worlds_1, self.inter_worlds_2):
               # print("self.input_worlds", len(self.input_worlds))
                res_emu = self.simulator.run_prog(cand_prog_inter_1, inp_world)
                if res_emu.status != 'OK' or res_emu.crashed:
                    # Crashed or failed the simulator
                    # Set the reward to negative and stop looking
                    # rew = self.reward_norm* 0.8
                    rew = -self.reward_norm
                    break
                elif res_emu.outgrid != inter_worlds_1:
                    # Generated a wrong state
                    # Set the reward to negative and stop looking
                    rew = self.reward_norm* 0.8 * 0.1
                    break
                else:
                    #res_emu = self.simulator.run_prog(cand_prog, inp_world)
                    #if res_emu.status != 'OK' or res_emu.crashed:
                        ## Crashed or failed the simulator
                        ## Set the reward to negative and stop looking
                        #rew = -self.reward_norm *0.1
                        #break
                    #elif res_emu.outgrid != out_world:
                        ## Generated a wrong state
                        ## Set the reward to negative and stop looking
                        #rew = -self.reward_norm *0.1
                        #break
                    #else:
                    rew = 0
                    
                    
                    parse_success_notimp, cand_prog_inter_2  = self.simulator.get_prog_ast(inter_trace_2)
                    if ((not parse_success) or (not parse_success_notimp)):
                        # Program is not syntactically correct
                        rew =  -self.reward_norm
                    else:
                        for inp_world, out_world, inter_worlds_1, inter_worlds_2 in zip(self.input_worlds, self.output_worlds, self.inter_worlds_1, self.inter_worlds_2):
                        # print("self.input_worlds", len(self.input_worlds))
                            res_emu = self.simulator.run_prog(cand_prog_inter_2, inp_world)
                            if res_emu.status != 'OK' or res_emu.crashed:
                                # Crashed or failed the simulator
                                # Set the reward to negative and stop looking
                                rew =   -self.reward_norm
                                break
                            elif res_emu.outgrid != inter_worlds_2:
                                # Generated a wrong state
                                # Set the reward to negative and stop looking
                                rew =  self.reward_norm* 0.9 * 0.1
                                break
                            else:
                                #res_emu = self.simulator.run_prog(cand_prog, inp_world)
                                #if res_emu.status != 'OK' or res_emu.crashed:
                                    ## Crashed or failed the simulator
                                    ## Set the reward to negative and stop looking
                                    #rew = -self.reward_norm *0.1
                                    #break
                                #elif res_emu.outgrid != out_world:
                                    ## Generated a wrong state
                                    ## Set the reward to negative and stop looking
                                    #rew = -self.reward_norm *0.1
                                    #break
                                #else:
                                rew = 0
                    
                    
                                parse_success_notimp, cand_prog_inter_3  = self.simulator.get_prog_ast(inter_trace_3)
                                if ((not parse_success) or (not parse_success_notimp)):
                                    # Program is not syntactically correct
                                    rew = -self.reward_norm
                                else:
                                    for inp_world, out_world, inter_worlds_1, inter_worlds_2 in zip(self.input_worlds, self.output_worlds, self.inter_worlds_1, self.inter_worlds_2):
                                    # print("self.input_worlds", len(self.input_worlds))
                                        res_emu = self.simulator.run_prog(cand_prog_inter_3, inp_world)
                                        if res_emu.status != 'OK' or res_emu.crashed:
                                            # Crashed or failed the simulator
                                            # Set the reward to negative and stop looking
                                            rew =  -self.reward_norm
                                            break
                                        elif res_emu.outgrid != out_world:
                                            # Generated a wrong state
                                            # Set the reward to negative and stop looking
                                            rew =  self.reward_norm* 1.0 * 0.1
                                            break
                                        else:
                                            #res_emu = self.simulator.run_prog(cand_prog, inp_world)
                                            #if res_emu.status != 'OK' or res_emu.crashed:
                                                ## Crashed or failed the simulator
                                                ## Set the reward to negative and stop looking
                                                #rew = -self.reward_norm *0.1
                                                #break
                                            #elif res_emu.outgrid != out_world:
                                                ## Generated a wrong state
                                                ## Set the reward to negative and stop looking
                                                #rew = -self.reward_norm *0.1
                                                #break
                                            #else:
                                            rew =  self.reward_norm
        return rew
    
def expected_rew_renorm(prediction_lpbs, prediction_reward_list):
    '''
    Simplest Reward Combination Function

    Takes as input:
    `prediction_lpbs`: The log probabilities of each sampled programs
    `prediction_reward_list`: The reward associated with each of these
                              sampled programs.

    Returns the expected reward under the (renormalized so that it sums to 1)
    probability distribution defined by prediction_lbps.
    '''
    # # Method 1:
    # pbs = prediction_lpbs.exp()
    # pb_sum = pbs.sum()
    # pbs = pbs.div(pb_sum.expand_as(pbs))

    # Method 2:
    prediction_pbs = F.softmax(prediction_lpbs, dim=0)

    if prediction_pbs.is_cuda:
        prediction_reward = torch.cuda.FloatTensor(prediction_reward_list)
    else:
        prediction_reward = torch.FloatTensor(prediction_reward_list)
    prediction_reward = Variable(prediction_reward, requires_grad=False)

    return torch.dot(prediction_pbs, prediction_reward)

EnvironmentClasses = {
    "BlackBoxGeneralization": MultiIOGrid,
    "BlackBoxConsistency": MultiIOGrid,
}
 
RewardCombinationFun = {
    "RenormExpected": expected_rew_renorm
}
