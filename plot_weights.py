# @Author: Fabian Kresse <fabian>
# @Date:   2021-06-10T08:47:38+02:00
# @Project: Aspp Accelerator
# @Filename: plot_weights.py
# @Last modified by:   fabian
# @Last modified time: 2021-07-16T12:10:05+02:00



import numpy as np
import matplotlib.pyplot as plt

#for i in range(9):
with open('data/weights_prunned0.'+str(5)+'.npy', 'rb') as f:
    weights = np.load(f)
    scale_weights = np.load(f)
    zero_points_kernels = np.load(f)

print(weights.shape)

zero_arr = []
for ofm in range(weights.shape[0]):
    for ifmap in range(0,weights.shape[1]-64,64):
        for x in range(3):
            for y in range(3):
                item = np.count_nonzero(weights[ofm,ifmap:ifmap+64,x,y]==0)
                zero_arr.append(item)

bs = max(zero_arr)-min(zero_arr)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots()

n, bins, patches = ax.hist(zero_arr, bs,edgecolor= 'black',linewidth=0.6, density=False)
fig.tight_layout()
ax.yaxis.grid(which="both",linestyle ='dashed')
plt.xlabel('None zero values per slice', fontsize=20)
plt.ylabel('Amount of slices', fontsize=20)
plt.savefig("../../thesis/BachelorThesis/figures/Results/weightdens.pdf",transparent=True)

plt.show()

#plot ifmp

with open('data/input_prunned0.'+str(5)+'6_0.npy', 'rb') as f:
    iacts = np.load(f)[0]
    scale_iacts = np.load(f)
    zero_points_iacts = np.load(f)

zero_arr_iacts = []
print(iacts.shape)
for ifmap in range(0,iacts.shape[0]-64,64):
        for x in range(33):
            for y in range(33):
                item = np.count_nonzero(iacts[ifmap:ifmap+64,x,y]==zero_points_iacts)
                zero_arr_iacts.append(item)

bs = max(zero_arr)-min(zero_arr)

fig, ax = plt.subplots()
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 20})
ax.yaxis.grid(which="both",linestyle ='dashed')
n, bins, patches = ax.hist(zero_arr_iacts, bs,edgecolor= 'black',linewidth=0.6, density=False)
fig.tight_layout()
#plt.savefig("../../thesis/BachelorThesis/figures/Results/weightdens.pdf",transparent=True)
plt.show()




ordering = []
for ofm in range(32):
#    print("new ofm")

    var_l = []
    for ifmap in range(0,64*10,64):
        item = 0

        for x in range(3):
            for y in range(3):
                item += np.count_nonzero(weights[ofm,ifmap:ifmap+64,x,y]==0)
                item /= 64
        var_l.append(item)
    #print(min(var_l))
    ordering.append(min(var_l))
fig, ax1 = plt.subplots(1, 1)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 20})
ax1.bar([i for i in range(len(ordering))],ordering)

ax1.yaxis.grid(which="both",linestyle ='dashed')
plt.savefig("../../thesis/BachelorThesis/figures/Results/weights_notreordered.pdf",transparent=True)
plt.show()



weights_subs = weights[0:32]
ordering = np.array(ordering)
arr1inds = ordering.argsort()


#print(arr1inds)
#print(weights.shape)
#static_indices = np.indices(weights.shape)
print("--------")
print(arr1inds)
weights_reorderd = weights_subs[arr1inds]
print("---")
for i in range(32):
    print(weights_reorderd[i,0,0,0])




ordering = []
for ofm in range(32):
#    print("new ofm")

    var_l = []
    for ifmap in range(0,64*10,64):
        item = 0

        for x in range(3):
            for y in range(3):
                item += np.count_nonzero(weights_reorderd[ofm,ifmap:ifmap+64,x,y]==0)
                item /= 64
        var_l.append(item)
    #print(min(var_l))
    ordering.append(min(var_l))
fig, ax1 = plt.subplots(1, 1)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 20})
ax1.bar([i for i in range(len(ordering))],ordering)

ax1.yaxis.grid(which="both",linestyle ='dashed')
plt.savefig("../../thesis/BachelorThesis/figures/Results/weights_reordered.pdf",transparent=True)
plt.show()
