# 
# coding: utf-8

# # Antenna Calibration
# 
# Flowchart of an algorithm to carry out the reference device calibration.
# <img src="images/main_workflow.png"style="width: 400px; height:600px">

# In[1]:


import numpy as np


# EDM matrix representation:
# <img src="images/EDM_matrix.png"style="width: 500px; height:100px">

# In[2]:


#hyper-parameters

#type in the EDM actual and EDM measure array row by row (dXY=0 when X=Y)
EDM_actual=np.array([[0,7.914,7.914],[7.914,0,7.914],[7.914,7.914,0]])
EDM_measure=np.array([[0,162.1613,162.2531],[162.1720,0,162.2449],[162.2155,162.2582,0]])

print("EDM Actual:\n"+str(EDM_actual))
print("EDM Measured:\n"+ str(EDM_measure)+ '\n')

ToF_actual=EDM_actual * 10/3  # in nano seconds(ns), using speed of light 3*10^8
ToF_measure=EDM_measure * 10/3

print("ToF Actual:\n"+str(ToF_actual))
print("ToF Measured:\n"+ str(ToF_measure)+'\n')


# In[3]:


N=EDM_actual.shape[0]  # number of device
print("number of device:", N)

iterations = 300 #number of iterations that the optimization algorithm runs
print("# of iterations:",iterations)

candidates_num=1000
print("number of random generalized candidates:",candidates_num)

mean=513 #ns
print("mean:",mean)
var=6    #ns
print("variance",var)

#initialize antenna_delay array, shape is (candidates_num, N), uniform distribution around mean and variance
antenna_delay_candidates= 513+np.random.uniform(-1*var,var,size=(candidates_num,N)) 
#print("initialize antenna_delay array: ", antenna_delay_candidates)


# Flowchart for populate and evaluate function.
# <table width=800,height=700>
#     <tr>
#         <td><img src="images/populate.png"style="width: 400px; height:600px"></td>
#         <td><img src="images/evaluate.png"style="width: 400px; height:600px"></td>
#     </tr>
# 
# </table>

# In[4]:


def populate(antenna_delay_candidates,perturbation):
    n_select=int(antenna_delay_candidates.shape[0]/4)
    new_candidates=antenna_delay_candidates[:n_select]
    
    final_candidates=np.array(new_candidates)
    for i in range(3):
        random_perturbation=new_candidates+np.random.uniform(-1*perturbation,perturbation,size=new_candidates.shape)
        final_candidates=np.vstack([final_candidates,random_perturbation])
    return final_candidates

def evaluate(antenna_delay_candidates,ToF_actual, ToF_measure):
    
    N=antenna_delay_candidates.shape[0]
    score=np.zeros((N,))
    
    for k in range(antenna_delay_candidates.shape[0]):
        antenna_delay=antenna_delay_candidates[k]
        ToF_candidates=np.zeros(( ToF_measure.shape))
        for i in range(ToF_candidates.shape[0]):
            for j in range(ToF_candidates.shape[1]):
                if(i != j):
                    ToF_candidates[i][j]=-1/2*antenna_delay[i]-1/2*antenna_delay[j]+ToF_measure[i][j]
        score[k]=np.linalg.norm(ToF_actual-ToF_candidates,np.inf)
    
    idx=score.argsort()  #sort the score low to high
    return np.take(antenna_delay_candidates,idx,axis=0),score[idx[0]]
    


# In[5]:


perturbation=0.2 #ns
final_score=0
for i in range(iterations):
    if(i>0):
        if(i%20==0):
            perturbation/=2
        antenna_delay_candidates=populate(antenna_delay_candidates,perturbation)  
        
    antenna_delay_candidates,final_score=evaluate(antenna_delay_candidates,ToF_actual, ToF_measure)
    


# In[6]:


final_antenna_delay=antenna_delay_candidates[0]
print("antenna delay:",final_antenna_delay,"\nfinal norm score:",final_score)


# In[7]:


# EDM_actual=np.array([[0,7.914,7.914],[7.914,0,7.914],[7.914,7.914,0]])
# EDM_measure=np.array([[0,162.1613,162.2531],[162.1720,0,162.2449],[162.2155,162.2582,0]])

# antenna_delay=np.array([514.4747,514.5911,515.0413])
# ToF_candidates=EDM_measure

# for i in range(ToF_candidates.shape[0]):
#     for j in range(ToF_candidates.shape[1]):
#         if(i != j):
#             ToF_candidates[i][j]=-1/2*antenna_delay[i]-1/2*antenna_delay[j]+ToF_measure[i][j]
            
# print(np.linalg.norm(ToF_actual-ToF_candidates,np.inf))

