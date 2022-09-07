import numpy as np
import matplotlib.pyplot as plt
import math
# Using readlines()
file1 = open('test.json', 'r')
Lines = file1.readlines()
x = 0
logs_enabled = False

def get_first_common_element(branch, main , b_passed_index, m_passed_index):
    b_index = b_passed_index
    m_index = m_passed_index
    m_value = ""
    second_token_match = False
    for m_element in main[m_passed_index:]:
        b_index = b_passed_index
        for b_element in branch[b_passed_index:]:
            
            if (b_element == m_element):
                if(logs_enabled):
                    print("equal", b_element)
                m_value = b_element
                if((m_index < len(main)-1) and (b_index < len(branch)-1)):
                    if (main[m_index+1] == branch[b_index+1]):
                        second_token_match = True
                return m_value, b_index, m_index, second_token_match
            b_index += 1
        m_index+=1
        if(logs_enabled):
            print("m_index in func", m_index)
        
    return m_value, b_index, m_index, second_token_match

# def get_first_common_element(x,y, index):
    # ''' Fetches first element from x that is common for both lists
        # or return None if no such an element is found.
    # '''
    # y_index = x_index = ret_index = index
    # x_value = y_value = ""
    # for i in y[index:]:
        # if (i == x[index]):
            # y_value = i 
            # y_index = ret_index
            # break
        # ret_index+=1
   
    # ret_index = index    
    # for i in x[index:]:
        # if (i == y[index]):
            # x_value = i 
            # x_index = ret_index
            # break
        # ret_index+=1
        
    # if( y_index < x_index):
        # return x_value, x_index, "y"
    # else:
    #    return y_value, y_index, "x"


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
        print("**********NEW**************")
        longest = actions_all_samples.index(max(actions_all_samples, key=len))
        if(longest == 0):
            main_list = actions_all_samples[0]
            branch_list = actions_all_samples[1]
        else:
            main_list = actions_all_samples[1]
            branch_list = actions_all_samples[0]  
        print("main", main_list)
        print("branch", branch_list)
        
        list_full = []
        list1 = []
        list2 = []
        minLength = min(len(x) for x in actions_all_samples )   
        maxLength = max(len(x) for x in actions_all_samples )    
        #for index in range(0, minLength):
        m_index= 0
        b_index = 0
        temp_index = 0
        while((m_index <= (maxLength-1))):
       # while((temp_index<10) and ((m_index <= (maxLength-1)))):
            temp_index+=1
            if(b_index == (minLength)):
                print("m_index", m_index)
                list_full.append([main_list[m_index], "NULL"])
                m_index +=1
            else:
                if (main_list[m_index] == branch_list[b_index]):
                    #list1.append(main_list[index])
                    #list2.append(branch_list[index])
                    list_full.append(main_list[m_index])
                    m_index +=1
                    b_index +=1
                else:
                    if(logs_enabled):
                        print("start m index", m_index)
                        print("start b index", b_index)
                    m_value,b_ret_index, m_ret_index, sm  =    get_first_common_element(branch_list, main_list, b_index, m_index)
                    
                    m_v2,b_2, m_2, sm_2 =    get_first_common_element(main_list, branch_list, m_index, b_index)
                    if ((abs(b_2-m_2) < abs(b_ret_index - m_ret_index)) or (sm_2 is True)):
                        m_value,b_ret_index, m_ret_index  = m_v2,m_2, b_2
                        
                    if(logs_enabled):
                        print("m_value, m_ret_index, b_ret_index", m_value, m_ret_index, b_ret_index)
                    
                    
                    # if(m_index > b_index):
                        # for i in range(b_index, b_ret_index):
                            # list_full.append(["NULL", branch_list[i]])
                            # b_index +=1
                            
                    if ((b_ret_index > b_index) and (m_ret_index == m_index)):
                            for i in range(b_index, b_ret_index):
                                list_full.append(["NULL", branch_list[i]])
                                b_index +=1
                                
                    elif ((m_ret_index > m_index) and (b_ret_index == b_index)):
                            for i in range(m_index, m_ret_index):
                                list_full.append([main_list[i], "NULL"])
                                m_index +=1
                        
                    elif(m_ret_index == b_ret_index):
                        for i in range(m_index, m_ret_index):
                            list_full.append([main_list[i], branch_list[i]])
                            m_index +=1
                            b_index +=1
                    
                    
                    elif(m_ret_index > b_ret_index):
                       
                        for i in range(m_index, b_ret_index):
                            list_full.append([main_list[i], branch_list[i]])
                            m_index +=1
                            b_index +=1
                        #if(m_ret_index > m_index):
                        # for i in range(b_ret_index, m_ret_index):
                            # list_full.append([main_list[i], "NULL"])
                            # m_index +=1
                            
                    else:
                        for i in range(m_index, m_ret_index):
                            list_full.append([main_list[i], branch_list[i]])
                            m_index +=1
                            b_index +=1
                        # for i in range(m_ret_index, b_ret_index):
                            # list_full.append(["NULL", branch_list[i]])
                            # b_index +=1
                            
                    
                
                
                # if(b_index == m_index):
                    # for i in range(index, m_index):
                        # list_full.append([main_list[i], branch_list[i]])
                    # index = m_index
        print("list_full", list_full)
            
              
                
        
        
    # if(number_of_traces == 2):
        # shortest = actions_all_samples.index(min(actions_all_samples, key=len))
        # print("full 1", actions_all_samples)
        # for x in actions_all_samples:
            # print((x))
        # maxLength = max(len(x) for x in actions_all_samples )
        # #print("max len", maxLength)
        
        # for index in range(0, maxLength):
            # if (actions_all_samples[0][index] == actions_all_samples[1][index]):
                # main_list.append(actions_all_samples[0][index])
            # else:
            
                # element, ret_index = check_for_if_else(actions_all_samples[0], actions_all_samples[1], index)
                # print("if else", element, ret_index)
                
                # print("ret_index before", index)
                # element, ret_index, to_fill =    get_first_common_element(actions_all_samples[0], actions_all_samples[1], index)
                # print("ele", element)
                # print("ret_index after", ret_index)
                # print("to_fill", to_fill)
                # if(to_fill == "x"):
                    # for j in range(index, ret_index):
                        # actions_all_samples[0].insert(index, "NULL")
                        # print("these indices to add", j)
                  # #  print("intermediate", actions_all_samples[0])
                    
                # if(to_fill == "y"):
                    # for j in range(index, ret_index):
                        # actions_all_samples[1].insert(index, "NULL")
                        # print("these indices to add", j)
                  # #  print("intermediate", actions_all_samples[0])
                
         #   print("index", index)
        list1 = actions_all_samples[0]
        list2 = actions_all_samples[1]
        # prints the missing and additional elements in list2 
        index = 0
        
     #   print("full", actions_all_samples)
     #   print("list1", actions_all_samples[0])
     #   print("list2", actions_all_samples[1])
        
        
     #   if ( ret_index ! = index
    #    print("first common", )
        
        
        
        
        
       # print( shortest)
        # for action in actions_all_samples:
            
            # main_list = [element for element in action if element in main_list]
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
    print(x)
    if( x>150):
        write_program(actions_per_sample)
    number_of_traces = len(actions_per_sample)
    count[number_of_traces-1] += 1
    
    actions_per_sample = []
   
    
    if(x == 201):
        break
# for action in actions_per_sample:
    # print(action)
# for actions_per_sample in actions_all_samples:
    # print(actions_per_sample)
    
    
#print("count", count)
    



        
