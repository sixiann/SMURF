#!/usr/bin/env python
# coding: utf-8

# # Tensorflow function

# In[104]:


import numpy as np
import tensorflow as tf
from scipy.ndimage import map_coordinates
from tensorflow_addons.image import interpolate_bilinear
import scipy.io as io
from matplotlib import pyplot as plt


# - Flow is stored in tensorflow as [batch_size, H, W, 2] or list([H, W, 2]) from https://github.com/philferriere/tfoptflow
# ![Screen%20Shot%202022-10-20%20at%206.28.35%20PM.png](attachment:Screen%20Shot%202022-10-20%20at%206.28.35%20PM.png)
# 
# 
# - dataloader of SMURF: 
#     * simple_dataset() class that only holds image triplet 
#     * def triplet_generator - """A generator that yields image triplets."""
#     * https://github.com/google-research/google-research/blob/03e5f5d01c4adee19c6064ded0f39783c573ea5b/smurf/data/simple_dataset.py#L30 
#     * https://github.com/google-research/google-research/blob/03e5f5d01c4adee19c6064ded0f39783c573ea5b/smurf/multiframe_training/main.py#L49
#     

# In[155]:


a = io.loadmat('/Users/darthvaper/Documents/flow_example.mat')


# In[243]:


flow1 = tf.convert_to_tensor(a['flow_01'], dtype = tf.float32)
flow1 = tf.expand_dims(flow1, axis = 0)

flow2 = tf.convert_to_tensor(a['flow_12'], dtype = tf.float32)
flow2 = tf.expand_dims(flow2, axis = 0)


import time 

start = time.time()
for i in range(10000):
    composed_flow = compose_flow(flow1, flow2)
end = time.time()
composed_flow = (composed_flow - np.nanmin(composed_flow)) / (np.nanmax(composed_flow) - np.nanmin(composed_flow))
end-start


# In[202]:


ref_flow = a['flow_02']
ref_flow = tf.convert_to_tensor(ref_flow, dtype = tf.float32)
ref_flow = tf.expand_dims(ref_flow, axis = 0)
ref_flow = (ref_flow - np.nanmin(ref_flow)) / (np.nanmax(ref_flow) - np.nanmin(ref_flow))
ref_flow.shape


# In[236]:


vis1 = tf.concat((ref_flow[0], tf.zeros(shape = (ref_flow[0].shape[0], ref_flow[0].shape[1], 1))), axis = 2)

plt.imshow(vis1)


# In[239]:


vis2 = tf.concat((composed_flow[0], tf.zeros(shape = (composed_flow[0].shape[0], composed_flow[0].shape[1], 1))), axis = 2)

plt.imshow(np.sum(np.abs((vis2-vis1)), axis = 2))


# In[1]:


def compose_flow(flow1, flow2):
    batch_size = flow1.shape[0]

    h,w = flow1.shape[1], flow2.shape[2]
    x,y = tf.meshgrid(tf.range(0,w, dtype = tf.float32), tf.range(0,h, dtype = tf.float32))

    grid1 = tf.stack([x,y], axis = 2)
    grid1 = tf.expand_dims(grid1, axis = 0)
    grid1 = tf.repeat(grid1, repeats = flow1.shape[0], axis = 0)


    query1 = tf.reshape(grid1+flow2, shape = (batch_size, h*w, 2))
    u = tf.where((query1[:,:,0] < flow1.shape[2]), query1[:,:,0], float('nan') )
    u = tf.where((u > 0), u, float('nan'))
    v = tf.where((query1[:,:,1] < flow1.shape[1]), query1[:,:,1], float('nan') )
    v = tf.where((v > 0), v, float('nan'))
    query1 = tf.stack([u,v], axis = 2)

    temp1 = interpolate_bilinear(grid1,query1, indexing = 'xy')
    u = tf.where((temp1[:,:,0] < flow1.shape[2]), temp1[:,:,0], float('nan') )
    u = tf.where((u > 0), u, float('nan'))
    v = tf.where((temp1[:,:,1] < flow1.shape[1]), temp1[:,:,1], float('nan') )
    v = tf.where((v > 0), v, float('nan'))
    temp1 = tf.stack([u,v], axis = 2)

    grid2 = tf.reshape(temp1, shape = (batch_size,h,w,2)) 
    query2 = tf.reshape(grid1 + flow1, shape = (batch_size, h*w, 2))
    u = tf.where((query2[:,:,0] < flow1.shape[2]), query2[:,:,0], float('nan') )
    u = tf.where((u > 0), u, float('nan'))
    v = tf.where((query2[:,:,1] < flow1.shape[1]), query2[:,:,1], float('nan') )
    v = tf.where((v > 0), v, float('nan'))
    query2 = tf.stack([u,v], axis = 2)
    
    
    temp2 = interpolate_bilinear(grid2,query2, indexing = 'xy')
    u = tf.where((temp2[:,:,0] < flow1.shape[2]), temp2[:,:,0], float('nan') )
    u = tf.where((u > 0), u, float('nan'))
    v = tf.where((temp2[:,:,1] < flow1.shape[1]), temp2[:,:,1], float('nan') )
    v = tf.where((v > 0), v, float('nan'))
    temp2 = tf.stack([u,v], axis = 2)
    grid3 = tf.reshape(temp2, shape = (batch_size,h,w,2))
    
    
    return grid3 - grid1


# In[152]:


compose_flow(flow1, flow2)


# # Numpy function

# In[324]:


flow1 = np.array([[[2,1], [0,0], [0,0]],
              [[0,0], [0,0], [0,0]],
              [[0,0], [1,-2], [0,0]]], dtype=np.float32)

flow2  = np.array([[[0,0],[0,0],[0,2]],
              [[0,0],[0,0], [-1,0]],
              [[0,0],[0,0],[0,0]]], dtype=np.float32)


# In[347]:


h,w = flow1.shape[0], flow2.shape[1]
x,y = np.meshgrid(np.arange(1,w+1), np.arange(1, h+1))
temp1 = map_coordinates(x.T, [x+flow2[:,:,0]-1, y+flow2[:,:,1]-1], order=3, mode='nearest').reshape(y.shape)
temp2 = map_coordinates(y.T, [(x+flow2[:,:,0]-1), (y+flow2[:,:,1]-1)], order=3, mode='nearest').reshape(x.shape)
temp1 = map_coordinates(temp1.T, [x+flow1[:,:,0]-1, y+flow1[:,:,1]-1], order=3, mode='nearest').reshape(y.shape)
temp2 = map_coordinates(temp2.T, [x+flow1[:,:,0]-1, y+flow1[:,:,1]-1], order=3, mode='nearest').reshape(y.shape)
composed = np.zeros((h,w,2))
composed[:,:,0] = temp1 - x
composed[:,:,1] = temp2 - y
composed


# # function

# In[368]:


def ComposeFlow2(F1,F2):
    indices = np.indices((F1.shape[0], F1.shape[1]))
    x = indices[0]
    y = indices[1]
    indices = np.stack((x,y), axis = -1) #indices = matrix with initial coordinates 
    
    row = indices.shape[0]
    col = indices.shape[1]
    coll = indices.shape[2]
    
    im2 = indices + np.flip(F1, axis = -1) #im2 = matrix with coordinates after F1 
    idx = (im2[:,:,0] * row + im2[:,:, 1]).flatten() 
    
    F22 = F2.reshape(row*col,coll)[idx].reshape(row,col,coll) 
    F22 = np.flip(F22, axis = -1) #F22 = F2 arranged with flow associated to each pixel 
    
    F13 = np.flip(((F22 + im2) - indices), axis = -1) 
    
    return F13
    


# In[373]:


ComposeFlow2(F1,F2)


# In[371]:


def ComposeFlow(F1,F2):
    F3 = np.empty(shape = (F1.shape[0], F1.shape[1]), dtype=tuple)

    def onepixel(x):
        x_im1_coord = x
        x_im2_coord = x_im1_coord + np.flip(F1[x_im1_coord[0], x_im1_coord[1]])
        x_im3_coord = x_im2_coord + np.flip(F2[x_im2_coord[0], x_im2_coord[1]])
        F13 = np.flip(x_im3_coord - x_im1_coord)

        return tuple(F13)
    
    #how to optimize this? 
    for i in range(F1.shape[0]):
        for j in range(F1.shape[1]):
            F3[i,j] = onepixel([i,j])
            
    #methods found online do not apply for functions that involve [i,j] as parameters 
    
    return F3
    


# In[372]:


ComposeFlow(F1,F2)

