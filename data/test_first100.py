import numpy as np
import matplotlib.pyplot as plt
import math
# Using readlines()
file1 = open('test.json', 'r')
Lines = file1.readlines()
x = 0

def check_for_if_else(x,y, index):
    y_index = ret_index = index
    y_value = ""
    for i in x[index:]:
        if (i == y[ret_index]):
            y_value = i 
            y_index = ret_index
            break
        ret_index+=1
        
    return y_value, y_index

def get_first_common_element(x,y, index):
    ''' Fetches first element from x that is common for both lists
        or return None if no such an element is found.
    '''
    y_index = x_index = ret_index = index
    x_value = y_value = ""
    for i in y[index:]:
        if (i == x[index]):
            y_value = i 
            y_index = ret_index
            break
        ret_index+=1
   
    ret_index = index    
    for i in x[index:]:
        if (i == y[index]):
            x_value = i 
            x_index = ret_index
            break
        ret_index+=1
        
    if( y_index < x_index):
        return x_value, x_index, "y"
    else:
        return y_value, y_index, "x"


def write_program(actions_all_samples):
    token_beg = ["DEF", "run", "m("] 
    token_end = ["m)"]
    
    prog = []
    prog += (token_beg)
    
    
   
    
    count = [ 0,0,0,0,0]
   
    main_list = []
    number_of_traces = len(actions_all_samples)
    count[number_of_traces-1] += 1
    
    if(number_of_traces == 1):
        prog = prog + actions_all_samples[0]
    if(number_of_traces == 2):
        shortest = actions_all_samples.index(min(actions_all_samples, key=len))
        print("full 1", actions_all_samples)
        for x in actions_all_samples:
            print((x))
        maxLength = max(len(x) for x in actions_all_samples )
        #print("max len", maxLength)
        
        for index in range(0, maxLength):
            if (actions_all_samples[0][index] == actions_all_samples[1][index]):
                main_list.append(actions_all_samples[0][index])
            else:
            
                element, ret_index = check_for_if_else(actions_all_samples[0], actions_all_samples[1], index)
                print("if else", element, ret_index)
                
                print("ret_index before", index)
                element, ret_index, to_fill =    get_first_common_element(actions_all_samples[0], actions_all_samples[1], index)
                print("ele", element)
                print("ret_index after", ret_index)
                print("to_fill", to_fill)
                if(to_fill == "x"):
                    for j in range(index, ret_index):
                        actions_all_samples[0].insert(index, "NULL")
                        print("these indices to add", j)
                  #  print("intermediate", actions_all_samples[0])
                    
                if(to_fill == "y"):
                    for j in range(index, ret_index):
                        actions_all_samples[1].insert(index, "NULL")
                        print("these indices to add", j)
                  #  print("intermediate", actions_all_samples[0])
                
         #   print("index", index)
        list1 = actions_all_samples[0]
        list2 = actions_all_samples[1]
        # prints the missing and additional elements in list2 
        index = 0
        
        print("full", actions_all_samples)
        print("list1", actions_all_samples[0])
        print("list2", actions_all_samples[1])
        
        
     #   if ( ret_index ! = index
    #    print("first common", )
        
        
        
        
        
       # print( shortest)
        for action in actions_all_samples:
            
            main_list = [element for element in action if element in main_list]
            #print("indv_list", action)
        #print("main_list", main_list)
    
    
    
    #for action in actions_all_samples:
        #main_list = [element for element in action if element in main_list]
        #print("indv_list", action)
    #print("main_list", main_list)
        
        
    prog += (token_end)
 #   print("prog", prog)
        
actions_per_sample = []
action_per_io = []
actions_all_samples = []
count = [ 0,0,0,0,0]
for line in Lines:
    start = 0
    end = 0
    for i in range(0,5):
        start = line.index('["',end+1)
        end = line.index(']',start+1)
        substring = line[start+1:end]
        y = substring.split(",")
        for action in y:
            action = action.replace('"', '')
            action = action.replace(' ', '')
            action_per_io.append(action)
        
        
        add_to_list = True
        for action in actions_per_sample:
            if action == action_per_io:
                add_to_list = False
        if(add_to_list):
            actions_per_sample.append(action_per_io)
        action_per_io = []
            
            
    actions_all_samples.append(actions_per_sample)
    x += 1
  #  print(x)
    write_program(actions_per_sample)
    number_of_traces = len(actions_per_sample)
    count[number_of_traces-1] += 1
    
    actions_per_sample = []
   
    
    if(x == 5):
        break
# for action in actions_per_sample:
    # print(action)
# for actions_per_sample in actions_all_samples:
    # print(actions_per_sample)
    
    
#print("count", count)
    



        
