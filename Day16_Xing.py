#!/usr/bin/env python
# coding: utf-8

# ## 전국 횡단 보도 표준 데이터 
# 
# 1. 횡단보도 연장, 녹색신호시간: 상관도
# 
# 2. 자전거 횡단도 겸용 비율(전체대비)
# 
# 3. 차로수별 자전거 횡단도 카운트/비율
# 
# 4. 차로수별 보행자 신호등 유무 카운트/ 비율 
# 
# 5. 차로수별 음향신호기 설치 유무 카운트/ 비율
# 
# ** 화면에 츌력
# 1. 상관도 수치
# 2. 비율 수치
# 3. 카운트 비율 
# 

# In[16]:


import numpy as np
import math
np.set_printoptions(precision=5, suppress=5)

def mean(num_list):
    return sum(num_list)/len(num_list)

def mean_2(num_list):
    sum_value = 0
    for i in num_list:
        sum_value = sum_value + i
    return sum_value / len(num_list)


def median(num_list):
    num_list.sort()
    if len(num_list)%2 ==1: 

        #몫만 가지고 계산할 경우 +1, -1을 굳이 안해줘도 됨 
        i = (len(num_list))//2
        return num_list[i]
    else:
        i = len(num_list)//2
        return (num_list[i]+ num_list[i-1])/2


def mean(number_list):
    return sum(number_list)/len(number_list)

def dev(num_list): #편차
    m= mean(num_list)
    return [x-m for x in num_list]

def ver(num_list): # 분산 #제곱
    n = len(num_list)
    d = dev(num_list)
    return sum([x*x for x in d])/(n-1)

def stdev(num_list): #표준편차
    return math.sqrt(ver(num_list))

def covar(list_1, list_2): #공분산 #따로 dev를 2개 구해서 짝을 이뤄서 곱해서 나눔 
    n = len(list_1)
    list_1_dev = dev(list_1)
    list_2_dev= dev(list_2)
    return sum(x * y for x, y in zip (list_1_dev, list_2_dev))/ (n-1)

def corel(list_1, list_2): # 상관도(-1 ~ 1) #상관도 : 두개의 변수의 서로의 상관관계, 영향도를 보여주는 것 
    return covar(list_1, list_2)/(stdev(list_1)*stdev(list_2))


# In[17]:


import numpy as np
import math
import matplotlib.pyplot as plt

def my_split(s):
    block_start = False
    start_index = 0
    ret_list=[]
    for i, c in enumerate(s):
        if block_start==False:
            if c==',':
                ret_list.append(s[start_index:i])
                start_index=i+1
            elif c=='"':
                block_start=True
                start_index = i
        else:
            if c=='"':
                block_start=False
    if s[-1]!=',':
        ret_list.append(s[start_index:])
    return ret_list
    
def split_len(data_list):
    len_list = []
    for e in data_list:
        len_list.append(len(e))
    print(set(len_list))
    if len (set(len_list))>1:
        for i in set(len_list):
            print(i, len_list.count(i))
    return set(len_list)

def check_split(list_data):
    t=set()
    for e in list_data:
        t.add(len(e))
    return len(t)


# In[18]:


csv_data = []
with open ('xing.csv') as f:
    for line in f:
        csv_data.append(my_split(line[:-1]))


for e in enumerate(csv_data[0]):
    print(e)
    
np_data = np.array(csv_data)


# In[19]:


#자전거 횡단도 겸용여부에 대한 비율 
btypes = np_data[1:, 7]
btypes_name, bytpes_counts = np.unique(btypes, return_counts = True)
print(btypes_name)
print(bytpes_counts)
print(bytpes_counts*100/ sum(bytpes_counts))


# In[20]:


a = bytpes_counts[2]/ sum(bytpes_counts[1:])

print(a*100)


# In[21]:


# 보행자 신호등 유무에 대한 비율 
btypes = np_data[1:, 14]
btypes = np.where(btypes == 'n','N', btypes)
btypes = np.where(btypes == 'y', 'Y', btypes)
btypes_name, bytpes_counts = np.unique(btypes, return_counts = True)
print(btypes_name)
print(bytpes_counts)
print(bytpes_counts*100/ sum(bytpes_counts))


# In[22]:


b = bytpes_counts[1]/ sum(bytpes_counts)
print(b*100)


# In[23]:


#음향신호기에 유무에 대한 비율 
btypes = np_data[1:, 16]
btypes = np.where(btypes=='n','N', btypes)
btypes_name, bytpes_counts = np.unique(btypes, return_counts = True)
print(btypes_name)
print(bytpes_counts)
print(bytpes_counts*100/ sum(bytpes_counts))



# In[24]:


c = bytpes_counts[2]/ sum(bytpes_counts[1:])
print(c*100)


# In[25]:


print(bytpes_counts[2])


# In[26]:


#my- split을 사용하여 데이터 정제 

def np_data_from_go_kr_csv(filename):
    t = []
    with open(filename, encoding='cp949') as f:
        for line in f:
            t.append(my_split(line[:-1]))
        if check_split(t)!= 1:
            return None
        else:
            return np.array(t)

def print_index_title(data):
    for e in enumerate(data[0]):
        print(e)




# In[27]:


if __name__ == '__main__':
    np_data = np_data_from_go_kr_csv('xing.csv')
#     print(target1(np_data))
#     print(target2(np_data))
#     print(target3(np_data))
#     print(target4(np_data))
#     print(target5(np_data))


# In[28]:


#1번 횡단보도 연장과 녹색신호시간의 상관도
def target1(data):
    sub_data = np_data[1:, [13, 17]]
    
    sub_data[sub_data =='0'] = ''
    #sub_data = np.where(sub_data == '0', '', sub_data)
    
    filter1 = sub_data[:,1]!=''
    sub_data_f = sub_data[filter1].astype(np.float64)
    #print(np.unique(sub_data[:,0]))
    #print(np.unique(sub_data[:,1]))
    #print(sub_data_f[:10])
    
    filter2 = sub_data_f[:,0] < 100
    sub_data_f2 = sub_data_f[filter2]

    _, axe = plt.subplots()
    axe.scatter(sub_data_f2[:,0], sub_data_f2[:,1])

    
    return np.corrcoef(sub_data_f[:,0], sub_data_f[:,1])

target1(np_data)


# In[29]:


def target2(data):
   sub_data = np_data[1:, 7]
   val, cnt = np.unique(sub_data, return_counts =True)
   #print(val, cnt)
   return cnt[2]*100/np.sum(cnt[1:])

target2(np_data)


# In[30]:


def target3(data):
    sub_data = np_data[1:, [11, 7]]
    #print(np.unique(sub_data[:, 0]))
    #print(np.unique(sub_data[:, 1]))
    t =[]
    for e in np.unique(sub_data[:,0]):
        filter1 = sub_data[:,0] ==e
        sub_data_f = sub_data[filter1]
        #print(sub_data_f)
        sub_data_f_y = sub_data_f[sub_data_f[:,1]=='Y']
        sub_data_f_n = sub_data_f[sub_data_f[:,1]== 'N']
        #print(e, len(sub_data_f_y), len(sub_data_f_n))
        #print(len(sub_data_f_y)/len(sub_data_f_n))
        yes_count = len(sub_data_f_y)
        no_count = len(sub_data_f_n)
        yes_no_count = yes_count+no_count
        if yes_no_count ==0:
            yes_no_count =1
        t.append((e, len(sub_data_f_y), len(sub_data_f_n), len(sub_data_f_y)/len(sub_data_f)))
       
    t = np.array(t).astype(np.float64)
    t = sorted(t, key=lambda x:x[0])
    return t

target3(np_data)


# In[31]:


def target4(data):
    sub_data = np_data[1:, [11, 14]]
    
    sub_data[sub_data =='n'] ='N'
    sub_data[sub_data =='y'] = 'Y'
    #print(np.unique(sub_data[:, 0]))
    #print(np.unique(sub_data[:, 1]))
    t =[]
    for e in np.unique(sub_data[:,0]):
        filter1 = sub_data[:,0] ==e
        sub_data_f = sub_data[filter1]
        #print(sub_data_f)
        sub_data_f_y = sub_data_f[sub_data_f[:,1]=='Y']
        sub_data_f_n = sub_data_f[sub_data_f[:,1]== 'N']
        #print(e, len(sub_data_f_y), len(sub_data_f_n))
        #print(len(sub_data_f_y)/len(sub_data_f_n))
        yes_count = len(sub_data_f_y)
        no_count = len(sub_data_f_n)
        yes_no_count = yes_count+no_count
        if yes_no_count ==0:
            yes_no_count =1
        t.append((e, len(sub_data_f_y), len(sub_data_f_n), len(sub_data_f_y)/len(sub_data_f)))
       
    t = np.array(t).astype(np.float64)
    t = sorted(t, key=lambda x:x[0])
    return t

target4(np_data)


# In[32]:


def target5(data):
    sub_data = np_data[1:, [11, 16]]
    
    sub_data[sub_data =='n'] ='N'
    sub_data[sub_data =='y'] = 'Y'
    #print(np.unique(sub_data[:, 0]))
    #print(np.unique(sub_data[:, 1]))
    t =[]
    for e in np.unique(sub_data[:,0]):
        filter1 = sub_data[:,0] ==e
        sub_data_f = sub_data[filter1]
        #print(sub_data_f)
        sub_data_f_y = sub_data_f[sub_data_f[:,1]=='Y']
        sub_data_f_n = sub_data_f[sub_data_f[:,1]== 'N']
        #print(e, len(sub_data_f_y), len(sub_data_f_n))
        #print(len(sub_data_f_y)/len(sub_data_f_n))
        yes_count = len(sub_data_f_y)
        no_count = len(sub_data_f_n)
        yes_no_count = yes_count+no_count
        if yes_no_count ==0:
            yes_no_count =1
        t.append((e, len(sub_data_f_y), len(sub_data_f_n), len(sub_data_f_y)/len(sub_data_f)))
       
    t = np.array(t).astype(np.float64)
    t = sorted(t, key=lambda x:x[0])
    return t

target5(np_data)

