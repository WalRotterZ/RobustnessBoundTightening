import os
import sys
import re
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import cvxpy as cp
import time
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed,wait
import onnx
import gurobipy as gp
from concurrent.futures import ThreadPoolExecutor, as_completed


def lpsolve(vars,cons,obj,solver=cp.GUROBI):
    prob=cp.Problem(obj,cons)
    prob.solve(solver=solver)
    return prob.value

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant) 
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant) 
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight        
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
        prev_abs_mode (int): indicates abstract mode of relu nodes in previous iteration.0 use first,1 use second
    """

    def __init__(self):
        self.algebra_lower=None
        self.algebra_upper=None
        self.concrete_algebra_lower=None
        self.concrete_algebra_upper=None
        self.concrete_lower=None
        self.concrete_upper=None
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.weight=None
        self.bias=None
        self.prev_abs_mode=None
        self.certain_flag=0
    
    def clear(self):
        self.certain_flag=0
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.prev_abs_mode=None

    def print(self):
        print('algebra_lower:',self.algebra_lower)
        print('algebra_upper:',self.algebra_upper)
        print('concrete_algebra_lower:',self.concrete_algebra_lower)
        print('concrete_algebra_upper:',self.concrete_algebra_upper)
        print('concrete_lower:',self.concrete_lower)
        print('concrete_upper:',self.concrete_upper)
        print('weight:',self.weight)
        print('bias:',self.bias)
        print('certain_flag:',self.certain_flag)

class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER=0
    AFFINE_LAYER=1
    RELU_LAYER=2    

    def __init__(self):
        self.size=None
        self.neurons=None
        self.layer_type=None
    
    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()

    def print(self):
        print('Layer size:',self.size)
        print('Layer type:',self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')

class network(object):
    
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
        unsafe_region (list of ndarray):coeffient of output and a constant
        property_flag (bool) : indicates the network have verification layer or not
        property_region (float) : Area of the input box
        abs_mode_changed (int) : count of uncertain relu abstract mode changed
        self.MODE_ROBUSTNESS=1
        self.MODE_QUANTITIVE=0
    """

    def __init__(self):
        self.MODE_QUANTITIVE=0
        self.MODE_ROBUSTNESS=1

        self.numlayers=None
        self.layerSizes=None
        self.inputSize=None
        self.outputSize=None
        self.mins=None
        self.maxes=None
        self.ranges=None
        self.layers=None
        self.property_flag=None
        self.property_region=None
        self.abs_mode_changed=None
        self.model=None
        self.lower_input_constraints=[]
        self.upper_input_constraints=[]
        self.dataset = None
        self.vnnlib = None
        self.netname = None
        self.magnification = 1
        self.ifSplit = False
        self.splitnum = 0
        self.splitcur = 0
        self.result_dir = "acasxuresult"
    
    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
    
    def verify_lp_split(self,PROPERTY,DELTA,MAX_ITER=5,SPLIT_NUM=0,WORKERS=12,TRIM=False,SOLVER=cp.GUROBI,MODE=0,USE_OPT_2=False):
        if SPLIT_NUM>self.inputSize:
            SPLIT_NUM=self.inputSize
        if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
        self.load_robustness(PROPERTY,DELTA,TRIM=TRIM)
        delta_list=[self.layers[0].neurons[i].concrete_upper-self.layers[0].neurons[i].concrete_lower for i in range(self.inputSize)]
        self.clear()
        self.deeppoly()
        verify_layer=self.layers[-1]
        verify_neuron_upper=np.array([neur.concrete_upper for neur in verify_layer.neurons])
        verify_list=np.argsort(verify_neuron_upper)
        for i in range(len(verify_list)):
            if verify_neuron_upper[verify_list[i]]>=0:
                verify_list=verify_list[i:]
                break
        if verify_neuron_upper[verify_list[0]]<0:
            print("Property Verified")
            if MODE==self.MODE_ROBUSTNESS:
                return True
            return
        split_list=[]        
        for i in range(2**SPLIT_NUM):
            cur_split=[]
            for j in range(SPLIT_NUM):
                if i & (2**j) == 0:
                    cur_split.append([self.layers[0].neurons[j].concrete_lower,(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2])
                else:
                    cur_split.append([(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2,self.layers[0].neurons[j].concrete_upper])
            for j in range(SPLIT_NUM,self.inputSize):
                cur_split.append([self.layers[0].neurons[j].concrete_lower,self.layers[0].neurons[j].concrete_upper])
            split_list.append(cur_split)
        
        obj=None
        prob=None
        constraints=None
        variables=[]
        for i in range(len(self.layers)):
            variables.append(cp.Variable(self.layers[i].size))
        unsafe_set=set()
        unsafe_set_deeppoly=set()
        unsafe_area_list=np.zeros(len(split_list))
        verified_list=[]
        verified_area=0
        for i in verify_list:
            verification_neuron=self.layers[-1].neurons[i]
            total_area=0
            for splits_num in range(len(split_list)):
                splits=split_list[splits_num]
                assert(len(splits)==self.inputSize)
                for j in range(self.inputSize):
                    self.layers[0].neurons[j].concrete_lower=splits[j][0]
                    self.layers[0].neurons[j].concrete_algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].concrete_upper=splits[j][1]
                    self.layers[0].neurons[j].concrete_algebra_upper=np.array([splits[j][1]])
                    self.layers[0].neurons[j].algebra_upper=np.array([splits[j][1]])            
                self.clear()
                for j in range(MAX_ITER):
                    self.deeppoly()
                    print("Abstract Mode Changed:",self.abs_mode_changed)
                    if (j==0) and (verification_neuron.concrete_upper>0):
                        unsafe_set_deeppoly.add(splits_num)
                    constraints=[]
                    #Build Constraints for each layer
                    for k in range(len(self.layers)):
                        cur_layer=self.layers[k]
                        cur_neuron_list=cur_layer.neurons
                        if cur_layer.layer_type==layer.INPUT_LAYER:
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]>=cur_neuron_list[p].concrete_lower)
                                constraints.append(variables[k][p]<=cur_neuron_list[p].concrete_upper)
                        elif cur_layer.layer_type==layer.AFFINE_LAYER:
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]==cur_neuron_list[p].weight@variables[k-1]+cur_neuron_list[p].bias)
                        elif cur_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==self.layers[k-1].size)
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]<=cur_neuron_list[p].algebra_upper[:-1]@variables[k-1]+cur_neuron_list[p].algebra_upper[-1])
                                # constraints.append(variables[k][p]>=cur_neuron_list[p].algebra_lower[:-1]@variables[k-1]+cur_neuron_list[p].algebra_lower[-1])
                                # Modified:using two lower bounds
                                constraints.append(variables[k][p]>=0)
                                constraints.append(variables[k][p]>=variables[k-1][p])
                    #Build the verification neuron constraint
                    for k in verified_list:
                        constraints.append(variables[-1][k]<=0)
                    #Modified:If MODE IS ROBUSTNESS AND USE_OPT_2 IS TRUE THEN CONSTRAINTS CAN BE ==0
                    if MODE==self.MODE_ROBUSTNESS and USE_OPT_2==True:
                        constraints.append(variables[-1][i]==0)
                    else:
                        constraints.append(variables[-1][i]>=0)
                    
                    #Check the feasibility
                    prob=cp.Problem(cp.Maximize(0),constraints)
                    prob.solve(solver=SOLVER)
                    if prob.status!=cp.OPTIMAL:                        
                        print("Split:",splits_num,"Infeasible")
                        break

                    #Refresh the input layer bounds
                    mppool=mp.Pool(WORKERS)
                    tasklist=[]
                    input_neurons=self.layers[0].neurons
                    for k in range(self.inputSize):
                        obj=cp.Minimize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                        obj=cp.Maximize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                    #Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    for k in range(self.inputSize):
                        if resultlist[k*2]>=input_neurons[k].concrete_lower:
                            input_neurons[k].concrete_lower=resultlist[k*2]
                            input_neurons[k].concrete_algebra_lower=np.array([resultlist[k*2]])
                            input_neurons[k].algebra_lower=np.array([resultlist[k*2]])
                        #
                        if resultlist[k*2+1]<=input_neurons[k].concrete_upper:
                            input_neurons[k].concrete_upper=resultlist[k*2+1]
                            input_neurons[k].concrete_algebra_upper=np.array([resultlist[k*2+1]])
                            input_neurons[k].algebra_upper=np.array([resultlist[k*2+1]])

                    #Refresh the uncertain ReLu's lowerbound
                    mppool=mp.Pool(WORKERS)
                    count=0
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    obj=cp.Minimize(variables[k][p])
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    if resultlist[index]>cur_layer.neurons[p].concrete_highest_lower:
                                        cur_layer.neurons[p].concrete_highest_lower=resultlist[index]
                                    if resultlist[index]>=0:
                                        next_layer.neurons[p].certain_flag=1
                                        count+=1
                                    index+=1

                    #Refresh the uncertain ReLu's upperbound
                    mppool=mp.Pool(WORKERS)
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    obj=cp.Maximize(variables[k][p])
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    if resultlist[index]<cur_layer.neurons[p].concrete_lowest_upper:
                                        cur_layer.neurons[p].concrete_lowest_upper=resultlist[index]
                                    if resultlist[index]<=0:
                                        next_layer.neurons[p].certain_flag=2
                                        count+=1
                                    index+=1
                    print('Refreshed ReLu nodes:',count)

                if prob.status==cp.OPTIMAL:
                    area=1
                    for j in range(self.inputSize):
                        area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/delta_list[j]
                    print("Split:",splits_num,"Area:",area*100)
                    if area>0:
                        if MODE==self.MODE_ROBUSTNESS:
                            return False
                        unsafe_area_list[splits_num]+=area
                        unsafe_set.add(splits_num)
                        total_area+=area
            print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',total_area*100)
            verified_area+=total_area
            verified_list.append(i)
        print('Overall Unsafe Overapproximate(Area)%',verified_area*100)
        verified_area=0
        for i in unsafe_area_list:
            if i>1/len(unsafe_area_list):
                verified_area+=1/len(unsafe_area_list)
            else:
                verified_area+=i
        print('Overall Unsafe Overapproximate(Smart Area)%',verified_area*100)
        print('Overall Unsafe Overapproximate(Box)%:',len(unsafe_set)/len(split_list)*100)
        print('Overall Unsafe Overapproximate(Deeppoly)%:',len(unsafe_set_deeppoly)/len(split_list)*100)
        if MODE==self.MODE_ROBUSTNESS:
            return True
        if MODE==self.MODE_QUANTITIVE:
            if verified_area<len(unsafe_set)/len(split_list):
                return [verified_area*100,len(unsafe_set_deeppoly)/len(split_list)*100]
            else:
                return [len(unsafe_set)/len(split_list)*100,len(unsafe_set_deeppoly)/len(split_list)*100]

    def deeppoly(self):

        def pre(cur_neuron,i):
            if i==0:
                cur_neuron.concrete_algebra_lower=deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper=deepcopy(cur_neuron.algebra_upper)
            lower_bound=deepcopy(cur_neuron.algebra_lower)
            upper_bound=deepcopy(cur_neuron.algebra_upper)
            for k in range(i+1)[::-1]:
                # print(k)
                tmp_lower=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert(self.layers[k].size+1==len(lower_bound))
                assert(self.layers[k].size+1==len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p]>=0:  
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_lower)                                 
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_lower
                    else:
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_upper)
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p]>=0:                        
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_lower                
                # print(tmp_lower)
                tmp_lower[-1]+=lower_bound[-1]
                tmp_upper[-1]+=upper_bound[-1]
                lower_bound=deepcopy(tmp_lower)
                upper_bound=deepcopy(tmp_upper)
                if k==1:
                    cur_neuron.concrete_algebra_upper=deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower=deepcopy(lower_bound)
            assert(len(lower_bound)==1)
            assert(len(upper_bound)==1)
            cur_neuron.concrete_lower=lower_bound[0]
            cur_neuron.concrete_upper=upper_bound[0]
            #add lowest and uppest history
            if (cur_neuron.concrete_highest_lower==None) or (cur_neuron.concrete_highest_lower<cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower=cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper==None) or (cur_neuron.concrete_lowest_upper>cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper=cur_neuron.concrete_upper            


        self.abs_mode_changed=0
        for i in range(len(self.layers)-1):
            # print('i=',i)
            pre_layer=self.layers[i]
            cur_layer=self.layers[i+1]
            pre_neuron_list=pre_layer.neurons
            cur_neuron_list=cur_layer.neurons
            if cur_layer.layer_type==layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    cur_neuron.algebra_lower=np.append(cur_neuron.weight,[cur_neuron.bias])
                    cur_neuron.algebra_upper=np.append(cur_neuron.weight,[cur_neuron.bias])
                    pre(cur_neuron,i)
            elif cur_layer.layer_type==layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    pre_neuron=pre_neuron_list[j]
                    #modified using lowest and uppest bound
                    if pre_neuron.concrete_highest_lower>=0 or cur_neuron.certain_flag==1:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        cur_neuron.algebra_upper[j]=1
                        cur_neuron.concrete_algebra_lower=deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper=deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower=pre_neuron.concrete_lower
                        cur_neuron.concrete_upper=pre_neuron.concrete_upper
                        #added
                        cur_neuron.concrete_highest_lower=pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper=pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag=1
                    elif pre_neuron.concrete_lowest_upper<=0 or cur_neuron.certain_flag==2:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)                        
                        cur_neuron.concrete_algebra_lower=np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper=np.zeros(self.inputSize)
                        cur_neuron.concrete_lower=0
                        cur_neuron.concrete_upper=0
                        #added
                        cur_neuron.concrete_highest_lower=0
                        cur_neuron.concrete_lowest_upper=0
                        cur_neuron.certain_flag=2
                    elif pre_neuron.concrete_highest_lower+pre_neuron.concrete_lowest_upper<=0:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=0):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=0

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)
                    else:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=1):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=1

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)

    def print(self):
        print('numlayers:%d' % (self.numLayers))
        print('layerSizes:',self.layerSizes)        
        print('inputSize:%d' % (self.inputSize))
        print('outputSize:%d' % (self.outputSize))
        print('mins:',self.mins)        
        print('maxes:',self.maxes)        
        print('ranges:',self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')
    
    def load_property(self, filename):
        self.property_flag=True
        self.property_region=1
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(x) for x in line.strip().split(' ')]
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)

    def load_robustness(self, filename,delta, TRIM=False):
        if self.property_flag==True:
                self.layers.pop()
                # self.clear()
        self.property_flag=True        
        with open(filename) as f:
            self.property_region=1
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(line.strip())-delta,float(line.strip())+delta]
                if TRIM:
                    if linedata[0]<0:
                        linedata[0]=0
                    if linedata[1]>1:
                        linedata[1]=1
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)

    def load_vnnlib(self, filename, magnification=1):
        lower_bound_pattern = re.compile(r'\(assert \(>= X_(\d+) ([\d\.eE+-]+)\)\)')
        upper_bound_pattern = re.compile(r'\(assert \(<= X_(\d+) ([\d\.eE+-]+)\)\)')
        lower_bounds, upper_bounds = [], []
        self.vnnlib = os.path.splitext(os.path.basename(filename))[0]
        self.magnification = magnification
        with open(filename, 'r') as file:
            for line in file:
                lower_match = lower_bound_pattern.search(line)
                if lower_match:
                    value = float(lower_match.group(2))
                    lower_bounds.append(value)
                upper_match = upper_bound_pattern.search(line)
                if upper_match:
                    value = float(upper_match.group(2))
                    upper_bounds.append(value)
        if self.dataset == "mnist":
            for lower, upper in zip(lower_bounds, upper_bounds):
                center = (lower + upper) / 2
                range_length = upper - lower
                new_lower = max(0, center - range_length * magnification)
                new_upper = min(1, center + range_length * magnification)
                self.lower_input_constraints.append(new_lower)
                self.upper_input_constraints.append(new_upper)
        elif self.dataset == "acasxu":
            for lower, upper in zip(lower_bounds, upper_bounds):
                center = (lower + upper) / 2
                range_length = upper - lower
                new_lower = center - range_length * magnification
                new_upper = center + range_length * magnification
                self.lower_input_constraints.append(new_lower)
                self.upper_input_constraints.append(new_upper)

    def load_nnet(self, filename):
        if "ACASXU" in filename:
            self.dataset = "acasxu"
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            #process the input layer
            self.layers=[]
            new_layer=layer()
            new_layer.layer_type=layer.INPUT_LAYER
            new_layer.size=layerSizes[0]
            new_layer.neurons=[]
            for i in range(layerSizes[0]):
                new_neuron=neuron()                
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.AFFINE_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    line=f.readline()
                    new_neuron=neuron()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    assert(len(aux)==previousLayerSize)
                    new_neuron.weight=np.array(aux)
                    new_layer.neurons.append(new_neuron)

                #biases                
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias=x
                
                self.layers.append(new_layer)                

                #add relu layer
                if layernum+1==numLayers:
                    break
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.RELU_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    new_neuron=neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges  
            self.property_flag=False

    def load_onnx(self,filename):
        if "mnist" in filename:
            self.dataset = "mnist"
        model = onnx.load(filename)
        graph = model.graph
        initializer = graph.initializer
        self.layers = []
        layersizes = []

        input_tensor = graph.input[0]
        input_size = input_tensor.type.tensor_type.shape.dim[1].dim_value
        self.inputSize = input_size
        layersizes.append(input_size)
        new_layer = layer()
        new_layer.layer_type = layer.INPUT_LAYER
        new_layer.size = layersizes[-1]
        new_layer.neurons = []
        for i in range(layersizes[-1]):
            new_neuron = neuron()
            new_layer.neurons.append(new_neuron)
        self.layers.append(new_layer)

        output_tensor = graph.output[0]
        output_size = output_tensor.type.tensor_type.shape.dim[1].dim_value
        self.outputSize = output_size

        count = 0
        for node in graph.node:
            if node.op_type == 'Gemm':
                new_layer = layer()
                new_layer.layer_type = layer.AFFINE_LAYER
                bias_data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer[2*count].data_type]
                bias_param_data = np.frombuffer(initializer[2*count].raw_data, dtype=bias_data_type).reshape(initializer[2*count].dims)
                weight_data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer[2*count+1].data_type]
                weight_param_data = np.frombuffer(initializer[2*count+1].raw_data, dtype=weight_data_type).reshape(initializer[2*count+1].dims)
                new_layer.size = weight_param_data.shape[0]
                layersizes.append(weight_param_data.shape[0])
                new_layer.neurons = []
                for i in range(layersizes[-1]):
                    new_neuron = neuron()
                    new_neuron.weight = weight_param_data[i]
                    new_neuron.bias = bias_param_data[i]
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)
                count += 1
            elif node.op_type == 'Relu':
                pre_size = layersizes[-1]
                layersizes.append(pre_size)
                new_layer = layer()
                new_layer.layer_type = layer.RELU_LAYER
                new_layer.size = pre_size
                new_layer.neurons = []
                for i in range(layersizes[-1]):
                    new_neuron = neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

        self.numLayers = len(self.layers) - 1
        self.layerSizes = layersizes

    def load_rlv(self,filename):
        layersize=[]
        dicts=[]        
        self.layers=[]
        with open(filename,'r') as f:
            line=f.readline()
            while(line):            
                if(line.startswith('#')):
                    linedata=line.replace('\n','').split(' ')
                    layersize.append(int(linedata[3]))
                    layerdict={}
                    if(linedata[4]=='Input'):
                        new_layer=layer()
                        new_layer.layer_type=layer.INPUT_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                            line=f.readline()
                            linedata=line.split(' ')
                            layerdict[linedata[1].replace('\n','')]=i
                        dicts.append(layerdict)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='ReLU'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                        #add relu layer
                        new_layer=layer()
                        new_layer.layer_type=layer.RELU_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='Linear') and (linedata[5]!='Accuracy'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                line=f.readline()
        self.layerSizes=layersize
        self.inputSize=layersize[0]
        self.outputSize=layersize[-1]
        self.numLayers=len(layersize)-1
        pass


    
    def itne_encoding_fnn(self,method=0,ep=0.001):

        model = gp.Model(name='ITNE_'+str(method))
        var, _var, delta_var, i = [], [], [], 0
        result_lower, result_upper = [], []
        for layer in self.layers:
            var.append([None]*len(layer.neurons))
            _var.append([None]*len(layer.neurons))
            delta_var.append([None]*len(layer.neurons))
        for layer in self.layers:
            if layer.layer_type == 0:
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=self.lower_input_constraints[j],ub=self.upper_input_constraints[j],name=f"input_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=self.lower_input_constraints[j],ub=self.upper_input_constraints[j],name=f"_input_neuron{i}{j+1}")
                    delta_var[i][j] = model.addVar(lb=-ep,ub=ep,name=f"delta_input_neuron{i}{j+1}")
                    model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                model.update()
                i += 1
            elif layer.layer_type == 1:
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=-gp.GRB.INFINITY,name=f"affine_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=-gp.GRB.INFINITY,name=f"_affine_neuron{i}{j+1}")
                    delta_var[i][j] = model.addVar(lb=-gp.GRB.INFINITY, name=f"delta_affine_neuron{i}{j+1}")
                model.update()
                for j in range(len(layer.neurons)):
                    assert len(layer.neurons[j].weight) == len(var[i-1])
                    expr = sum(layer.neurons[j].weight[m] * var[i-1][m] for m in range(len(var[i-1])))
                    _expr = sum(layer.neurons[j].weight[m] * _var[i-1][m] for m in range(len(_var[i-1])))
                    model.addConstr(var[i][j] == expr + layer.neurons[j].bias)
                    model.addConstr(_var[i][j] == _expr + layer.neurons[j].bias)
                    model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                model.update()
                i += 1
            elif layer.layer_type == 2:
                ub, _ub, lb, _lb = [None]*len(layer.neurons), [None]*len(layer.neurons), [None]*len(layer.neurons), [None]*len(layer.neurons)
                delta_ub, delta_lb = [None]*len(layer.neurons), [None]*len(layer.neurons)
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=0,name=f"relu_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=0,name=f"_relu_neuron{i}{j+1}")
                    delta_var[i][j] = model.addVar(lb=-gp.GRB.INFINITY, name=f"delta_relu_neuron{i}{j+1}")
                model.update()
                for j in range(len(layer.neurons)):
                    model.setObjective(var[i - 1][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of neuron{i - 1}{j + 1}")
                    ub[j] = var[i - 1][j].X
                    print(f"upper bound of neuron{i - 1}{j + 1} processed")
                    model.setObjective(var[i - 1][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of neuron{i - 1}{j + 1}")
                    lb[j] = var[i - 1][j].X
                    print(f"lower bound of neuron{i - 1}{j + 1} processed")
                    model.setObjective(_var[i - 1][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of _neuron{i - 1}{j + 1}")
                    _ub[j] = _var[i - 1][j].X
                    print(f"upper bound of _neuron{i - 1}{j + 1} processed")
                    model.setObjective(_var[i - 1][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of neuron{i - 1}{j + 1}")
                    _lb[j] = _var[i - 1][j].X
                    print(f"lower bound of _neuron{i - 1}{j + 1} processed")
                for k in range(len(layer.neurons)):
                    var[i - 1][k].setAttr('ub', ub[k])
                    var[i - 1][k].setAttr('lb', lb[k])
                    _var[i - 1][k].setAttr('ub', _ub[k])
                    _var[i - 1][k].setAttr('lb', _lb[k])
                model.update()
                for j in range(len(layer.neurons)):
                    lbi = min(var[i-1][j].lb,0)
                    ubi = max(var[i-1][j].ub,0)
                    diff = (ubi - lbi) if ubi - lbi > 0 else 1
                    model.addConstr(var[i][j] >= 0)
                    model.addConstr(var[i][j] >= var[i-1][j])
                    model.addConstr(var[i][j] <= ubi * (var[i-1][j] - lbi)/diff)
                    var[i][j].lb = max(var[i-1][j].lb, 0)
                    var[i][j].ub = max(var[i-1][j].ub, 0)
                    model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                    model.update()
                for j in range(len(layer.neurons)):
                    model.setObjective(delta_var[i-1][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of delta_neuron{i-1}{j+1}")
                    delta_ub[j] = delta_var[i-1][j].X
                    delta_var[i-1][j].ub = delta_ub[j]
                    model.update()
                    print(f"upper bound of delta_neuron{i - 1}{j + 1} processed")
                    model.setObjective(delta_var[i-1][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of delta_neuron{i-1}{j+1}")
                    delta_lb[j] = delta_var[i-1][j].X
                    delta_var[i-1][j].lb = delta_lb[j]
                    model.update()
                    print(f"lower bound of delta_neuron{i - 1}{j + 1} processed")
                    lbi = min(delta_var[i-1][j].lb,0)
                    ubi = max(delta_var[i-1][j].ub,0)
                    diff = (ubi - lbi) if ubi - lbi > 0 else 1
                    if method == 0:
                        if delta_var[i - 1][j].lb < 0 and delta_var[i - 1][j].ub > 0 and _lb[j] > 0:
                            #model.addConstr(delta_var[i][j] >= delta_var[i - 1][j])
                            #model.addConstr(delta_var[i][j] <= ubi * (delta_var[i - 1][j] - lbi) / diff)
                            model.addConstr(delta_var[i][j] == delta_var[i-1][j])
                        elif delta_var[i - 1][j].lb < 0 and delta_var[i - 1][j].ub > 0 and _ub[j] < 0:
                            #model.addConstr(delta_var[i][j] >= 0)
                            #model.addConstr(delta_var[i][j] <= ubi * (delta_var[i - 1][j] - lbi) / diff)
                            model.addConstr(delta_var[i][j] == 0)
                        else:
                            model.addConstr(delta_var[i][j] <= ubi * (delta_var[i - 1][j] - lbi) / diff)
                            model.addConstr(delta_var[i][j] >= lbi * (ubi - delta_var[i - 1][j]) / diff)
                    elif method == 1:
                        model.addConstr(delta_var[i][j] <= ubi * (delta_var[i - 1][j] - lbi) / diff)
                        model.addConstr(delta_var[i][j] >= lbi * (ubi - delta_var[i - 1][j]) / diff)
                    model.update()
                for j in range(len(layer.neurons)):
                    model.setObjective(delta_var[i][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of delta_neuron{i}{j+1}")
                    temp = delta_var[i][j].X
                    delta_var[i][j].ub = temp
                    model.update()
                    print(f"upper bound of delta_neuron{i}{j + 1} processed")
                    model.setObjective(delta_var[i][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of delta_neuron{i}{j+1}")
                    temp = delta_var[i][j].X
                    delta_var[i][j].lb = temp
                    model.update()
                    print(f"lower bound of delta_neuron{i}{j + 1} processed")
                i += 1
        if self.dataset == "mnist":
            out = len(self.layers) - 1
            for j in range(10):
                model.setObjective(delta_var[out][j], gp.GRB.MAXIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = delta_var[out][j].X
                    print(f"Upper Bound = {obj_opt}")
                    result_upper.append(obj_opt)
                model.setObjective(delta_var[out][j], gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = delta_var[out][j].X
                    print(f"Lower Bound = {obj_opt}")
                    result_lower.append(obj_opt)
            self.save_results(result_upper,result_lower,method)
        elif self.dataset == "acasxu":
            out = len(self.layers) - 1
            for j in range(5):
                model.setObjective(delta_var[out][j], gp.GRB.MAXIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = delta_var[out][j].X
                    print(f"Upper Bound = {obj_opt}")
                    result_upper.append(obj_opt)
                model.setObjective(delta_var[out][j], gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = delta_var[out][j].X
                    print(f"Lower Bound = {obj_opt}")
                    result_lower.append(obj_opt)
            self.save_results(result_upper, result_lower, method)



    '''
    def itne_encoding_fnn(self,method=0,ep=0.001):

        model = gp.Model(name='ITNE_'+str(method))
        var, _var, delta_var, i = [], [], [], 0
        result_lower, result_upper = [], []
        delta_var = [None]*len(self.layers[-1].neurons)
        for layer in self.layers:
            var.append([None]*len(layer.neurons))
            _var.append([None]*len(layer.neurons))
            #delta_var.append([None]*len(layer.neurons))
        for layer in self.layers:
            if layer.layer_type == 0:
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=self.lower_input_constraints[j],ub=self.upper_input_constraints[j],name=f"input_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=self.lower_input_constraints[j],ub=self.upper_input_constraints[j],name=f"_input_neuron{i}{j+1}")
                    #delta_var[i][j] = model.addVar(lb=-ep,ub=ep,name=f"delta_input_neuron{i}{j+1}")
                    #model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                model.update()
                i += 1
            elif layer.layer_type == 1:
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=-gp.GRB.INFINITY,name=f"affine_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=-gp.GRB.INFINITY,name=f"_affine_neuron{i}{j+1}")
                    #delta_var[i][j] = model.addVar(lb=-gp.GRB.INFINITY, name=f"delta_affine_neuron{i}{j+1}")
                model.update()
                for j in range(len(layer.neurons)):
                    assert len(layer.neurons[j].weight) == len(var[i-1])
                    expr = sum(layer.neurons[j].weight[m] * var[i-1][m] for m in range(len(var[i-1])))
                    _expr = sum(layer.neurons[j].weight[m] * _var[i-1][m] for m in range(len(_var[i-1])))
                    model.addConstr(var[i][j] == expr + layer.neurons[j].bias)
                    model.addConstr(_var[i][j] == _expr + layer.neurons[j].bias)
                    #model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                model.update()
                i += 1
            elif layer.layer_type == 2:
                ub, _ub, lb, _lb = [None]*len(layer.neurons), [None]*len(layer.neurons), [None]*len(layer.neurons), [None]*len(layer.neurons)
                delta_ub, delta_lb = [None]*len(layer.neurons), [None]*len(layer.neurons)
                for j in range(len(layer.neurons)):
                    var[i][j] = model.addVar(lb=0,name=f"relu_neuron{i}{j+1}")
                    _var[i][j] = model.addVar(lb=0,name=f"_relu_neuron{i}{j+1}")
                    #delta_var[i][j] = model.addVar(lb=-gp.GRB.INFINITY, name=f"delta_relu_neuron{i}{j+1}")
                model.update()
                for j in range(len(layer.neurons)):
                    model.setObjective(var[i - 1][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of neuron{i - 1}{j + 1}")
                    ub[j] = var[i - 1][j].X
                    print(f"upper bound of neuron{i - 1}{j + 1} processed")
                    model.setObjective(var[i - 1][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of neuron{i - 1}{j + 1}")
                    lb[j] = var[i - 1][j].X
                    print(f"lower bound of neuron{i - 1}{j + 1} processed")
                    model.setObjective(_var[i - 1][j], gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the maximum value of _neuron{i - 1}{j + 1}")
                    _ub[j] = _var[i - 1][j].X
                    print(f"upper bound of _neuron{i - 1}{j + 1} processed")
                    model.setObjective(_var[i - 1][j], gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status == gp.GRB.INFEASIBLE:
                        raise ValueError(f"Model is infeasible for the minimum value of neuron{i - 1}{j + 1}")
                    _lb[j] = _var[i - 1][j].X
                    print(f"lower bound of _neuron{i - 1}{j + 1} processed")
                for k in range(len(layer.neurons)):
                    var[i - 1][k].setAttr('ub', ub[k])
                    var[i - 1][k].setAttr('lb', lb[k])
                    _var[i - 1][k].setAttr('ub', _ub[k])
                    _var[i - 1][k].setAttr('lb', _lb[k])
                model.update()
                for j in range(len(layer.neurons)):
                    lbi = min(var[i-1][j].lb,0)
                    ubi = max(var[i-1][j].ub,0)
                    _lbi = min(_var[i-1][j].lb,0)
                    _ubi = max(_var[i-1][j].ub,0)
                    diff = (ubi - lbi) if ubi - lbi > 0 else 1
                    _diff = (_ubi - _lbi) if _ubi - _lbi > 0 else 1
                    model.addConstr(var[i][j] >= 0)
                    model.addConstr(var[i][j] >= var[i-1][j])
                    model.addConstr(var[i][j] <= ubi * (var[i-1][j] - lbi)/diff)
                    model.addConstr(_var[i][j] >= 0)
                    model.addConstr(_var[i][j] >= _var[i - 1][j])
                    model.addConstr(_var[i][j] <= _ubi * (_var[i - 1][j] - _lbi) / _diff)
                    var[i][j].lb = max(var[i-1][j].lb, 0)
                    var[i][j].ub = max(var[i-1][j].ub, 0)
                    _var[i][j].lb = max(_var[i-1][j].lb,0)
                    _var[i][j].ub = max(_var[i-1][j].ub,0)
                    #model.addConstr(delta_var[i][j] == var[i][j] - _var[i][j])
                    model.update()
                i += 1
        if self.dataset == "mnist":
            out = len(self.layers) - 1
            for j in range(10):
                delta_var[j] = model.addVar(lb=-gp.GRB.INFINITY,name=f"output_delta_neuron{j+1}")
                #model.addConstr(delta_var[j] == var[out][j] - _var[out][j])
                model.setObjective(var[out][j], gp.GRB.MAXIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = var[out][j].X
                    print(f"Upper Bound = {obj_opt}")
                    result_upper.append(obj_opt)
                model.setObjective(var[out][j], gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = var[out][j].X
                    print(f"Lower Bound = {obj_opt}")
                    result_lower.append(obj_opt)
            k_l = [lb - ub for lb, ub in zip(result_lower, result_upper)]
            k_u = [ub - lb for lb, ub in zip(result_lower, result_upper)]
            self.save_results(k_l,k_u,method)
        elif self.dataset == "acasxu":
            out = len(self.layers) - 1
            for j in range(5):
                delta_var[j] = model.addVar(lb=-gp.GRB.INFINITY, name=f"output_delta_neuron{j + 1}")
                #model.addConstr(delta_var[j] == var[out][j] - _var[out][j])
                model.setObjective(var[out][j], gp.GRB.MAXIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = var[out][j].X
                    print(f"Upper Bound = {obj_opt}")
                    result_upper.append(obj_opt)
                model.setObjective(var[out][j], gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    obj_opt = var[out][j].X
                    print(f"Lower Bound = {obj_opt}")
                    result_lower.append(obj_opt)
            k_l = [lb - ub for lb, ub in zip(result_lower, result_upper)]
            k_u = [ub - lb for lb, ub in zip(result_lower, result_upper)]
            self.save_results(k_l, k_u, method)
    '''

    def save_results(self,result_upper,result_lower,method):
        #result_dir = "result"
        #self.result_dir = "new_dimresult_our"
        os.makedirs(self.result_dir, exist_ok=True)
        if not self.ifSplit:
            filename_upper = f"{self.netname}_{self.vnnlib}_upper_x{self.magnification * 2}_method{method}.txt"
            filename_lower = f"{self.netname}_{self.vnnlib}_lower_x{self.magnification * 2}_method{method}.txt"
        else:
            filename_upper = f"splitnum{self.splitnum}_splitcur{self.splitcur}_{self.netname}_{self.vnnlib}_upper_x{self.magnification * 2}_method{method}.txt"
            filename_lower = f"splitnum{self.splitnum}_splitcur{self.splitcur}_{self.netname}_{self.vnnlib}_lower_x{self.magnification * 2}_method{method}.txt"
        filepath_upper = os.path.join(self.result_dir, filename_upper)
        filepath_lower = os.path.join(self.result_dir, filename_lower)
        with open(filepath_upper, "w") as f_upper:
            for value in result_upper:
                f_upper.write(f"{value}\n")
        with open(filepath_lower, "w") as f_lower:
            for value in result_lower:
                f_lower.write(f"{value}\n")

    def optimize_neuron(self,j,var,_var,model,i):
        result = {}

        # Maximizing var[i-1][j]
        model.setObjective(var[i - 1][j], gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status == gp.GRB.INFEASIBLE:
            raise ValueError(f"Model is infeasible for the maximum value of neuron{i - 1}{j + 1}")
        elif model.Status == gp.GRB.UNBOUNDED:
            raise ValueError(f"Model is unbounded for the maximum value of neuron{i - 1}{j + 1}")
        else:
            result['ub'] = var[i - 1][j].X
        print(f"Maximum Value of Neuron{i - 1}{j + 1} Processed")

        # Minimizing var[i-1][j]
        model.setObjective(var[i - 1][j], gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status == gp.GRB.INFEASIBLE:
            raise ValueError(f"Model is infeasible for the minimum value of neuron{i - 1}{j + 1}")
        elif model.Status == gp.GRB.UNBOUNDED:
            raise ValueError(f"Model is unbounded for the minimum value of neuron{i - 1}{j + 1}")
        else:
            result['lb'] = var[i - 1][j].X
        print(f"Minimum Value of Neuron{i - 1}{j + 1} Processed")

        # Maximizing _var[i-1][j]
        model.setObjective(_var[i - 1][j], gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status == gp.GRB.INFEASIBLE:
            raise ValueError(f"Model is infeasible for the maximum value of _neuron{i - 1}{j + 1}")
        elif model.Status == gp.GRB.UNBOUNDED:
            raise ValueError(f"Model is unbounded for the maximum value of _neuron{i - 1}{j + 1}")
        else:
            result['_ub'] = _var[i - 1][j].X
        print(f"Maximum Value of _Neuron{i - 1}{j + 1} Processed")

        # Minimizing _var[i-1][j]
        model.setObjective(_var[i - 1][j], gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status == gp.GRB.INFEASIBLE:
            raise ValueError(f"Model is infeasible for the minimum value of _neuron{i - 1}{j + 1}")
        elif model.Status == gp.GRB.UNBOUNDED:
            raise ValueError(f"Model is unbounded for the minimum value of _neuron{i - 1}{j + 1}")
        else:
            result['_lb'] = _var[i - 1][j].X
        print(f"Minimum Value of _Neuron{i - 1}{j + 1} Processed")


        return j,result


    def find_max_disturbance(self,PROPERTY,L=0,R=1000,TRIM=False):
        ans=0
        while L<=R:
            # print(L,R)
            mid=int((L+R)/2)
            self.load_robustness(PROPERTY,mid/1000,TRIM=TRIM)
            self.clear()
            self.deeppoly()
            flag=True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                ans=mid/1000
                L=mid+1
            else:
                R=mid-1
        return ans
    
    def find_max_disturbance_lp(self,PROPERTY,L,R,TRIM,WORKERS=12,SOLVER=cp.GUROBI):
        ans=L
        while L<=R:
            mid=int((L+R)/2)
            if self.verify_lp_split(PROPERTY=PROPERTY,DELTA=mid/1000,MAX_ITER=5,SPLIT_NUM=0,WORKERS=WORKERS,TRIM=TRIM,SOLVER=SOLVER,MODE=1):
                print("Disturbance:",mid/1000,"Success!")
                ans=mid/1000
                L=mid+1
            else:
                print("Disturbance:",mid/1000,"Failed!")
                R=mid-1
        return ans


def process_mnist_onnx_vnnlib(onnx_folder, vnnlib_folder, onnx_pairs,m=1):
    execution_time_dir = 'new_executiontime_our'
    if not os.path.exists(execution_time_dir):
        os.makedirs(execution_time_dir)
    for onnx_model, vnnlib_list in onnx_pairs:
        onnx_path = os.path.join(onnx_folder, onnx_model)
        for vnnlib_file in vnnlib_list:
            vnnlib_path = os.path.join(vnnlib_folder, vnnlib_file)
            net = network()
            net.netname = onnx_model
            net.load_onnx(onnx_path)
            net.load_vnnlib(vnnlib_path,magnification=m)
            start_time_net = time.time()
            net.itne_encoding_fnn(method=0)
            end_time_net = time.time()
            net_execution_time = end_time_net - start_time_net
            net_time_filename = f"{onnx_model}_{net.vnnlib}_x{net.magnification}_method0_time.txt"
            net_time_path = os.path.join(execution_time_dir, net_time_filename)
            with open(net_time_path, 'w') as f:
                f.write(f"net execution time: {net_execution_time:.4f} seconds\n")
            '''
            _net = network()
            _net.netname = onnx_model
            _net.load_onnx(onnx_path)
            _net.load_vnnlib(vnnlib_path,magnification=m)
            start_time__net = time.time()
            _net.itne_encoding_fnn(method=1)
            end_time__net = time.time()
            _net_execution_time = end_time__net - start_time__net
            _net_time_filename = f"{onnx_model}_{_net.vnnlib}_x{_net.magnification}_method1_time.txt"
            _net_time_path = os.path.join(execution_time_dir, _net_time_filename)
            with open(_net_time_path, 'w') as f:
                f.write(f"_net execution time: {_net_execution_time:.4f} seconds\n")
            '''



'''
def process_acasxu_nnet_vnnlib(nnet_folder, vnnlib_folder, nnet_pairs,m=2):
    execution_time_dir = 'new_dimexecutiontime_our'
    if not os.path.exists(execution_time_dir):
        os.makedirs(execution_time_dir)

    def split_dimension(lower, upper):
        mid = (lower + upper) / 2
        return [(lower, mid), (mid, upper)]

    for nnet_model, vnnlib_list in nnet_pairs:
        nnet_path = os.path.join(nnet_folder, nnet_model)
        for vnnlib_file in vnnlib_list:
            vnnlib_path = os.path.join(vnnlib_folder, vnnlib_file)

            net = network()
            net.netname = nnet_model
            net.load_nnet(nnet_path)
            net.load_vnnlib(vnnlib_path,magnification=m)

            net.lower_input_constraints[1] = max(net.lower_input_constraints[1], -0.5)
            net.upper_input_constraints[1] = min(net.upper_input_constraints[1], 0.5)
            net.lower_input_constraints[2] = max(net.lower_input_constraints[2], -0.5)
            net.upper_input_constraints[2] = min(net.upper_input_constraints[2], 0.5)
            first_dim_bounds = split_dimension(net.lower_input_constraints[0], net.upper_input_constraints[0])
            second_dim_bounds = split_dimension(net.lower_input_constraints[1], net.upper_input_constraints[1])
            third_dim_bounds = split_dimension(net.lower_input_constraints[2], net.upper_input_constraints[2])
            fourth_dim_bounds = split_dimension(net.lower_input_constraints[3], net.upper_input_constraints[3])
            fifth_dim_bounds = split_dimension(net.lower_input_constraints[4], net.upper_input_constraints[4])


            # dim 145 start
            # dim 1
            subspaces_2 = []
            for first_lower, first_upper in first_dim_bounds:
                subspace = (first_lower, first_upper)
                subspaces_2.append(subspace)
            for i, (first_lower, first_upper) in enumerate(subspaces_2):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = max(net_sub.lower_input_constraints[1], -0.5), min(net_sub.upper_input_constraints[1], 0.5)
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 2
                net_sub.splitcur = i
                net_sub.result_dir = "dim145result"
                net_sub.itne_encoding_fnn(method=0)



            # dim1 and dim4
            subspaces_4 = []
            for first_lower, first_upper in first_dim_bounds:
                for fourth_lower, fourth_upper in fourth_dim_bounds:
                    subspace = (first_lower, first_upper, fourth_lower, fourth_upper)
                    subspaces_4.append(subspace)

            for i, (first_lower, first_upper, fourth_lower, fourth_upper) in enumerate(subspaces_4):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[3], net_sub.upper_input_constraints[3] = fourth_lower, fourth_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = max(
                    net_sub.lower_input_constraints[1], -0.5), min(net_sub.upper_input_constraints[1], 0.5)
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(
                    net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 4
                net_sub.splitcur = i
                net_sub.result_dir = "dim145result"
                net_sub.itne_encoding_fnn(method=0)

            # dim 1 and dim 4 and dim 5
            subspaces_8 = []
            for first_lower, first_upper in first_dim_bounds:
                for fourth_lower, fourth_upper in fourth_dim_bounds:
                    for fifth_lower, fifth_upper in fifth_dim_bounds:
                        subspace = (first_lower, first_upper, fourth_lower, fourth_upper, fifth_lower, fifth_upper)
                        subspaces_8.append(subspace)

            for i, (first_lower, first_upper, fourth_lower, fourth_upper, fifth_lower, fifth_upper) in enumerate(
                    subspaces_8):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[3], net_sub.upper_input_constraints[3] = fourth_lower, fourth_upper
                net_sub.lower_input_constraints[4], net_sub.upper_input_constraints[4] = fifth_lower, fifth_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = max(
                    net_sub.lower_input_constraints[1], -0.5), min(net_sub.upper_input_constraints[1], 0.5)
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(
                    net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 8
                net_sub.splitcur = i
                net_sub.result_dir = "dim145result"
                net_sub.itne_encoding_fnn(method=0)


            # dim 123 start

            # dim 1
            subspaces_start_with_dim_1 = []
            for first_lower, first_upper in first_dim_bounds:
                subspace = (first_lower, first_upper)
                subspaces_start_with_dim_1.append(subspace)
            for i, (first_lower, first_upper) in enumerate(subspaces_start_with_dim_1):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = max(
                    net_sub.lower_input_constraints[1], -0.5), min(net_sub.upper_input_constraints[1], 0.5)
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(
                    net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 2
                net_sub.splitcur = i
                net_sub.result_dir = "dim123result"
                net_sub.itne_encoding_fnn(method=0)

            # dim1 and dim2
            subspaces_1_2 = []
            for first_lower, first_upper in first_dim_bounds:
                for second_lower, second_upper in second_dim_bounds:
                    subspace = (first_lower, first_upper, second_lower, second_upper)
                    subspaces_1_2.append(subspace)

            for i, (first_lower, first_upper, second_lower, second_upper) in enumerate(subspaces_1_2):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = second_lower, second_upper
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(
                    net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 4
                net_sub.splitcur = i
                net_sub.result_dir = "dim123result"
                net_sub.itne_encoding_fnn(method=0)

            # dim1 and dim2 and dim3
            subspaces_1_2_3 = []
            for first_lower, first_upper in first_dim_bounds:
                for second_lower, second_upper in second_dim_bounds:
                    for third_lower, third_upper in third_dim_bounds:
                        subspace = (first_lower, first_upper, second_lower, second_upper, third_lower, third_upper)
                        subspaces_1_2_3.append(subspace)

            for i, (first_lower, first_upper, second_lower, second_upper, third_lower, third_upper) in enumerate(
                    subspaces_1_2_3):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[0], net_sub.upper_input_constraints[0] = first_lower, first_upper
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = second_lower, second_upper
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = third_lower, third_upper
                net_sub.ifSplit = True
                net_sub.splitnum = 8
                net_sub.splitcur = i
                net_sub.result_dir = "dim123result"
                net_sub.itne_encoding_fnn(method=0)


            # dim 235 start

            # dim 2
            subspaces_start_with_dim_2 = []
            for second_lower, second_upper in second_dim_bounds:
                subspace = (second_lower, second_upper)
                subspaces_start_with_dim_2.append(subspace)

            for i, (second_lower, second_upper) in enumerate(subspaces_start_with_dim_2):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = second_lower, second_upper
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = max(
                    net_sub.lower_input_constraints[2], -0.5), min(net_sub.upper_input_constraints[2], 0.5)
                net_sub.ifSplit = True
                net_sub.splitnum = 2
                net_sub.splitcur = i
                net_sub.result_dir = "dim235result"
                net_sub.itne_encoding_fnn(method=0)

            # dim2 and dim3
            subspaces_2_3 = []
            for second_lower, second_upper in second_dim_bounds:
                for third_lower, third_upper in third_dim_bounds:
                    subspace = (second_lower, second_upper, third_lower, third_upper)
                    subspaces_2_3.append(subspace)

            for i, (second_lower, second_upper, third_lower, third_upper) in enumerate(subspaces_2_3):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = second_lower, second_upper
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = third_lower, third_upper
                net_sub.ifSplit = True
                net_sub.splitnum = 4
                net_sub.splitcur = i
                net_sub.result_dir = "dim235result"
                net_sub.itne_encoding_fnn(method=0)

            # dim2 and dim3 and dim5
            subspaces_2_3_5 = []
            for second_lower, second_upper in second_dim_bounds:
                for third_lower, third_upper in third_dim_bounds:
                    for fifth_lower, fifth_upper in fifth_dim_bounds:
                        subspace = (second_lower, second_upper, third_lower, third_upper, fifth_lower, fifth_upper)
                        subspaces_2_3_5.append(subspace)

            for i, (second_lower, second_upper, third_lower, third_upper, fifth_lower, fifth_upper) in enumerate(
                    subspaces_2_3_5):
                net_sub = network()
                net_sub.netname = nnet_model
                net_sub.load_nnet(nnet_path)
                net_sub.load_vnnlib(vnnlib_path, magnification=m)
                net_sub.lower_input_constraints[1], net_sub.upper_input_constraints[1] = second_lower, second_upper
                net_sub.lower_input_constraints[2], net_sub.upper_input_constraints[2] = third_lower, third_upper
                net_sub.lower_input_constraints[4], net_sub.upper_input_constraints[4] = fifth_lower, fifth_upper
                net_sub.ifSplit = True
                net_sub.splitnum = 8
                net_sub.splitcur = i
                net_sub.result_dir = "dim235result"
                net_sub.itne_encoding_fnn(method=0)

            # non split

            net.ifSplit = True
            net.splitnum = 0
            net.splitcur = 0
            net.itne_encoding_fnn(method=0)
'''




def process_acasxu_nnet_vnnlib(nnet_folder, vnnlib_folder, nnet_pairs,m=2):
    execution_time_dir = 'new_executiontime_our'
    if not os.path.exists(execution_time_dir):
        os.makedirs(execution_time_dir)
    for nnet_model, vnnlib_list in nnet_pairs:
        nnet_path = os.path.join(nnet_folder, nnet_model)
        for vnnlib_file in vnnlib_list:
            vnnlib_path = os.path.join(vnnlib_folder, vnnlib_file)
            net = network()
            net.netname = nnet_model
            net.load_nnet(nnet_path)
            net.load_vnnlib(vnnlib_path,magnification=m)
            start_time_net = time.time()
            net.itne_encoding_fnn(method=0)
            end_time_net = time.time()
            net_execution_time = end_time_net - start_time_net
            net_time_filename = f"{nnet_model}_{net.vnnlib}_x{net.magnification}_method0_time.txt"
            net_time_path = os.path.join(execution_time_dir, net_time_filename)
            with open(net_time_path, 'w') as f:
                f.write(f"net execution time: {net_execution_time:.4f} seconds\n")
            '''
            _net = network()
            _net.netname = nnet_model
            _net.load_nnet(nnet_path)
            _net.load_vnnlib(vnnlib_path,magnification=m)
            start_time__net = time.time()
            _net.itne_encoding_fnn(method=1)
            end_time__net = time.time()
            _net_execution_time = end_time__net - start_time__net
            _net_time_filename = f"{nnet_model}_{_net.vnnlib}_x{_net.magnification}_method1_time.txt"
            _net_time_path = os.path.join(execution_time_dir, _net_time_filename)
            with open(_net_time_path, 'w') as f:
                f.write(f"_net execution time: {_net_execution_time:.4f} seconds\n")
            '''
            



if __name__ == "__main__":
    onnx_folder = "onnx"
    nnet_folder = "nnet"
    vnnlib_folder = "vnnlib"
    '''
    net = network()
    net.load_nnet("nnet/ACASXU_experimental_v2a_1_1.nnet")
    net.load_vnnlib("vnnlib/acasxu_prop_4.vnnlib",magnification=2)
    start_time_net = time.time()
    net.itne_encoding_fnn(method=0)
    end_time_net = time.time()
    net_execution_time = end_time_net - start_time_net
    print(net_execution_time)
'''
    #process_mnist_onnx_vnnlib(onnx_folder, vnnlib_folder)
    #process_acasxu_nnet_vnnlib(nnet_folder, vnnlib_folder)


