#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1：
#(A)
def reverseStr(s):
    r_s=s[::-1]
    
    return r_s
#(B)
def newReverse(s):
    s=s.split(" ")
    y=""
    for x in s:
        y=y+reverseStr(x)+" "
    return y


# In[2]:


#Q2：
def numProblem(x):
    result=[]
    for a in range(1,x+1):
        
        if(a%3!=0 and a%5!=0  or a%15==0 or a==15):
            result.append(a)
        
    return len(result)

numProblem(30)      


# In[37]:


#Q3：
'''
查看混和的袋子，因標籤一定錯，所以可推論標籤為混和袋子內只有一支筆，
而當我們確認是哪種筆時，此時在另一個標籤為另一種筆的袋中就能確認是混和的筆，並進一步推論出第三個袋子中的筆是沒出現過的筆，eg.當我們從標為混和的袋中拿出原子筆，
那麼標為鉛筆的袋子中不可能是鉛筆(與標示相符)，也不可能是原子筆(已出現了)，因此可以推論出
裡面是混和的筆，而此時也就確認標為原子筆中裝的是鉛筆，反之亦然
'''
#Q4：
'''
270*3指的是三人目前實際所付出的金錢，然而服務生私吞的60元卻不是指
服務生從他們應該付出的錢(即750)中暗中拿走的，因此把兩者相加後得到的數字870是沒有意義的，應該是以
60+750=810(他們實際付的錢)才對
'''


# In[ ]:




