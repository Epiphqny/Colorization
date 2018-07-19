import numpy as np
import json

# f=open('feature.txt','r')
# names=[]
# feas=[]
# i=0
# for line in f:
# 	name,feature=json.loads(line)
# 	names.append(name)
# 	feas.append(np.asarray(feature))
#
# 	i+=1
# 	if i%100==0:
# 		print(i)
# max=0
# np.save('names.npy',arr=np.asarray(names))
# arr=np.zeros((4500,4500))
# for i in range(4500):
# 	if i%100==0:
# 		print(i)
# 	for j in range(i+1,4500):
# 		arr[i][j]=arr[j][i]=np.mean(feas[i]*feas[j])
# np.save('sim.npy',arr)
#
arr=np.load('../data/sim_conv7_max.npy')
# # print(arr)
# # print(np.max(arr))
size=arr.shape[0]
result=[]
f=open('../data/ind_conv7_max.txt','w')
max_list=[]
max_val=0
for i in range(size):
    #if i%100==0:
    #    print(i)
    ind=[]
    #print(arr[i])
    #print(sorted(arr[i]))
    #max_list.append(max(arr[i]))
    #for j in range(size):
    #if arr[i][j]>0.0039:
    
    b=np.argwhere(arr[i]>0.001953125)
    b.reshape(b.shape[0])
    ind=b.tolist()
    #print(ind)
    #ind.append(j)
    f.write('%s\n' % json.dumps(ind, ensure_ascii = False))
    if len(ind)>max_val:
        max_val=len(ind)
    print(len(ind))
f.close()
print('max_val',max_val)

#print(sorted(max_list,reverse = True))

'''
f=open('data/ind_conv8_max.txt','r')
names=np.load('names_conv8_max.npy')
i=0
for line in f:
	id_list=json.loads(line)
	if len(id_list)>100:
		f_write = open(str(i)+'.txt', 'w')
		for id in id_list:
			print(id)
			f_write.write(names[id]+'\n')
	i+=1

f_write.close()
'''


