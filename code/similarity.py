import numpy as np
import json

f=open('../data/conv7_avg.txt','r')
names=[]
feas=[]
i=0
for line in f:
    name,feature=json.loads(line)
    names.append(name)
    feas.append(np.asarray(feature))
    #print(feature)
    i+=1
    if i%100==0:
        print(i)
    #if i==100:
    #    break
max=0
size=len(names)
test=np.load('../data/test_conv7.npy')
#np.save('../data/conv7_avg_names.npy',arr=np.asarray(names))
#arr=np.zeros((size,size))
res=[]
for i in range(size):
    res.append(np.mean(test*feas[i]))
    #if i%10==0:
        #print(i)
    #for j in range(i,size):
        #a=feas[i]/np.max(feas[i])
        #b=feas[j]/np.max(feas[j])
        #print('i',arr[i])
        #print('j',arr[j])
        #arr[i][j]=arr[j][i]=np.mean(feas[i]*feas[j])
        #print(arr[i][j])
        #arr[i][j]=arr[j][i]=((feas[i]-feas[j])**2).mean()
        #print(arr[i][j])
np.save('../data/sim_array_test.npy',np.asarray(res))
#np.save('../data/sim_conv7_avg.npy',arr)
