'''
This module provides a set of high-level neural networks layers.
'''

import tensorflow as tf
from functools import reduce

from utils import get_transformation_matrix_shape
from utils import euclidean_norm
from ops import routing


def fully_connected(inputs, activation,
                    num_outputs,
                    out_caps_shape,
                    routing_method='EMRouting',
                    num_iter=3,
                    reuse=None):
    '''A capsule fully connected layer.
    Args:

        inputs: A tensor with shape [batch_size, num_inputs] + in_caps_shape.
        activation: [batch_size, num_inputs]
        num_outputs: Integer, the number of output capsules in the layer.
        out_caps_shape: A list with two elements, pose shape of output capsules.
    Returns:
        pose: [batch_size, num_outputs] + out_caps_shape
        activation: [batch_size, num_outputs]
    '''
    in_pose_shape = inputs.get_shape().as_list()
    num_inputs = in_pose_shape[1] # here the num_inputs = height*width*filter, which has been done beforing sending into this full_connected layer
    batch_size = in_pose_shape[0]
    T_size = get_transformation_matrix_shape(in_pose_shape[-2:], out_caps_shape) # return [shape1[0],shape0[0]]: [16,8]
    T_shape = [1, num_inputs, num_outputs] + T_size
    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape, initializer=tf.random_normal_initializer(mean=0, stddev=1))
    T_b      = tf.get_variable("transformation_b", shape=[1, num_inputs, num_outputs]+out_caps_shape, initializer=tf.random_normal_initializer(mean=0, stddev=1))
    T_b      = tf.tile(T_b, [batch_size, 1, 1, 1, 1])
    T_matrix = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1]) # repeat the matrix to: [batch_size, input_num, output_num ] + cap_trans_shape
    inputs = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, num_outputs, 1, 1]) # in the shape: [batch_size, input_num, output_num]+ input_cap_shape
    with tf.variable_scope('transformation'):
        # vote: [batch_size, num_inputs, num_outputs] + out_caps_shape
        vote = tf.matmul(T_matrix, inputs) + T_b
        # print('vote_shape'+vote.shape)
    with tf.variable_scope('routing'):
        if routing_method == 'EMRouting':
            activation = tf.reshape(activation, shape=activation.get_shape().as_list() + [1, 1])
            vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs, -1])
            pose, activation = routing(vote, activation, num_outputs, out_caps_shape, routing_method, num_iter=num_iter)
            pose = tf.reshape(pose, shape=[batch_size, num_outputs] + out_caps_shape)
            activation = tf.reshape(activation, shape=[batch_size, -1])
        elif routing_method == 'DynamicRouting':
            pose, _ = routing(vote, activation, num_outputs=num_outputs, out_caps_shape=out_caps_shape, method=routing_method, num_iter=num_iter)
            pose = tf.squeeze(pose, axis=1)
            activation = tf.squeeze(euclidean_norm(pose))
    return(pose, activation)


def primaryCaps(input, filters,
                kernel_size,
                strides,
                out_caps_shape,
                padding='VALID',
                method=None,
                regularizer=None):
    '''PrimaryCaps layer
    Args:
        input: [batch_size, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space.
        kernel_size: ...
        strides: ...
        out_caps_shape: ... # how many caps and the dimension of each output capsule. e.g.[8 1]
        method: the method of calculating probability of entity existence(logistic, norm, None)
    Returns:
        pose: [batch_size, out_height, out_width, filters] + out_caps_shape
        activation: [batch_size, out_height, out_width, filters]
    '''
    # pose matrix
    pose_size = reduce(lambda x, y: x * y, out_caps_shape)
    pose = tf.layers.conv2d(input, filters * pose_size,
                            kernel_size=kernel_size,
                            strides=strides, activation=None,
                            activity_regularizer=regularizer, padding=padding)
    pose_shape = pose.get_shape().as_list()[:3] + [filters] + out_caps_shape
    pose = tf.reshape(pose, shape=pose_shape)

    if method == 'logistic':
        # logistic activation unit
        activation = tf.layers.conv2d(input, filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=tf.nn.sigmoid,
                                      activity_regularizer=regularizer)
    elif method == 'norm':
        activation = euclidean_norm(pose)
    else:
        activation = None

    return(pose, activation)

def PrimaryCaps_multi(inputs, filters,
                kernel_size,
                strides,
                out_caps_shape,
                padding='SAME',
                method='norm',
                regularizer=None):
    '''PrimaryCaps layer
    Args:
        input:       a list of tensors. [batch_size, in_height, in_width, in_channels].
        filters:     a list of Integer, the dimensionality of the output space for each tensor.
        kernel_size: a list of Integer, the kernel size for each input tensor
        strides:     a list of Integer, strides for each input tensor
        out_caps_shape: ... # how many caps and the dimension of each output capsule. e.g.[8 1]
        method: the method of calculating probability of entity existence(logistic, norm, None)
    Returns:
        pose: [batch_size, num_total_caps] + out_caps_shape
        activation: [batch_size, num_total_caps]
    '''
    # pose matrix
    assert len(inputs) == len(filters)
    assert len(inputs) == len(kernel_size)
    assert len(inputs) == len(strides)
    in_num = len(inputs)
    pose_size = reduce(lambda x, y: x * y, out_caps_shape) # reduce is to 
    out_pose = []
    for i in range(in_num):
        pose = tf.layers.conv2d(inputs[i], filters[i] * pose_size,
                                kernel_size=kernel_size[i],
                                strides=strides[i], activation=None,
                                activity_regularizer=regularizer, padding=padding)
        # pose_shape = pose.get_shape().as_list()[:3] + [filters[i]] + out_caps_shape
        pose_shape = [inputs[i].get_shape().as_list()[0], -1] + out_caps_shape
        pose = tf.reshape(pose, shape=pose_shape)
        if len(tf.shape(out_pose).get_shape().as_list()) < 2:
            out_pose = pose
        else:
            out_pose = tf.concat([out_pose, pose], axis=1)

    # if method == 'logistic':
    #     # logistic activation unit
    #     activation = tf.layers.conv2d(inputs[i], filters[i],
    #                                   kernel_size=kernel_size,
    #                                   strides=strides,
    #                                   activation=tf.nn.sigmoid,
    #                                   activity_regularizer=regularizer)
    if method == 'norm':
        activation = euclidean_norm(pose)
    else:
        activation = None

    return(pose, activation)

def conv2d(in_pose,
           activation,
           filters,
           out_caps_shape,
           kernel_size,
           strides=(1, 1),
           coordinate_addition=False,
           regularizer=None,
           reuse=None):
    '''A capsule convolutional layer.
    Args:
        in_pose: A tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_shape.
        activation: A tensor with shape [batch_size, in_height, in_width, in_channels]
        filters: ...
        out_caps_shape: ...
        kernel_size: ...
        strides: ...
        coordinate_addition: ...
        regularizer: apply regularization on a newly created variable and add the variable to the collection tf.GraphKeys.REGULARIZATION_LOSSES.
        reuse: ...
    Returns:
        out_pose: A tensor with shape [batch_size, out_height, out_height, out_channals] + out_caps_shape,
        out_activation: A tensor with shape [batch_size, out_height, out_height, out_channels]
    '''
    # do some preparation stuff
    in_pose_shape = in_pose.get_shape().as_list()
    in_caps_shape = in_pose_shape[-2:]
    batch_size = in_pose_shape[0]
    in_channels = in_pose_shape[3]

    T_size = get_transformation_matrix_shape(in_caps_shape, out_caps_shape)
    if isinstance(kernel_size, int):
        h_kernel_size = kernel_size
        w_kernel_size = kernel_size
    elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
        h_kernel_size = kernel_size[0]
        w_kernel_size = kernel_size[1]
    if isinstance(strides, int):
        h_stride = strides
        w_stride = strides
    elif isinstance(strides, (list, tuple)) and len(strides) == 2:
        h_stride = strides[0]
        w_stride = strides[1]
    num_inputs = h_kernel_size * w_kernel_size * in_channels
    batch_shape = [batch_size, h_kernel_size, w_kernel_size, in_channels]
    T_shape = (1, num_inputs, filters) + tuple(T_size)

    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape, regularizer=regularizer)
    T_matrix_batched = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])

    h_step = int((in_pose_shape[1] - h_kernel_size) / h_stride + 1)
    w_step = int((in_pose_shape[2] - w_kernel_size) / w_stride + 1)
    out_pose = []
    out_activation = []
    # start to do capsule convolution.
    # Note: there should be another way more computationally efficient to do this
    for i in range(h_step):
        col_pose = []
        col_prob = []
        h_s = i * h_stride
        h_e = h_s + h_kernel_size
        for j in range(w_step):
            with tf.variable_scope("transformation"):
                begin = [0, i * h_stride, j * w_stride, 0, 0, 0]
                size = batch_shape + in_caps_shape
                w_s = j * w_stride
                pose_sliced = in_pose[:, h_s:h_e, w_s:(w_s + w_kernel_size), :, :, :]
                pose_reshaped = tf.reshape(pose_sliced, shape=[batch_size, num_inputs, 1] + in_caps_shape)
                shape = [batch_size, num_inputs, filters] + in_caps_shape
                batch_pose = tf.multiply(pose_reshaped, tf.constant(1., shape=shape))
                vote = tf.reshape(tf.matmul(T_matrix_batched, batch_pose), shape=[batch_size, num_inputs, filters, -1])
                # do Coordinate Addition. Note: not yet completed
                if coordinate_addition:
                    x = j / w_step
                    y = i / h_step

            with tf.variable_scope("routing") as scope:
                if i > 0 or j > 0:
                    scope.reuse_variables()
                begin = [0, i * h_stride, j * w_stride, 0]
                size = [batch_size, h_kernel_size, w_kernel_size, in_channels]
                prob = tf.slice(activation, begin, size)
                prob = tf.reshape(prob, shape=[batch_size, -1, 1, 1])
                pose, prob = routing(vote, prob, filters, out_caps_shape, method="EMRouting", regularizer=regularizer)
            col_pose.append(pose)
            col_prob.append(prob)
        col_pose = tf.concat(col_pose, axis=2)
        col_prob = tf.concat(col_prob, axis=2)
        out_pose.append(col_pose)
        out_activation.append(col_prob)
    out_pose = tf.concat(out_pose, axis=1)
    out_activation = tf.concat(out_activation, axis=1)

    return(out_pose, out_activation)

'''
PrimaryCaps - reshape - full_connected caps - reshape - dePrimaryCaps
            (2d capsules to 1d)             (1d capsules to 2d)
This part of code(reshape) is done in the net building, not implemented here
'''
def dePrimaryCaps(inputs, activation,
                  num_outputs,
                  kernel_size=4,
                  strides=2,
                  padding='same'):
    ''' dePrimaryCaps: inverse layer of PrimaryCaps. 
        PrimaryCaps layer did two thing: 1. conv 2. reshape to capsule
        dePrimaryCaps will do two thing: 1. reshape from 2d capsule feature map to normal CNN layer
                                         2. deconv
    Args:
        input: [batch_size, num_total_caps] + in_caps_shape
        activation: [batch_size, num_total_caps]
        kernel_size: ... used for conv
        strides: ... used for conv
    Returns:
        CNN layer: [batch_size, H, W, C]
    ----------------------------------
    This is the first version of dePrimaryCaps layer, but we found the performance was not very good. 
    We redesigned the dePrimaryCaps layer. 
    The first version 
    '''
    in_pose_shape = inputs.get_shape().as_list()
    caps_shape = in_pose_shape[-2:]
    batch_size = in_pose_shape[0]
    pose_h = in_pose_shape[1]
    pose_w = in_pose_shape[2]

    # caps_size = reduce(lambda x, y: x * y, caps_shape)
    pose = tf.reshape(inputs, shape=[batch_size,  pose_h, pose_w, -1])
    # the pose is reshape into the normal CNN layer, at this time, C = caps_c * caps_size
    # below we do the deconv 
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(pose, num_outputs, kernel_size=kernel_size, strides=(strides,strides), \
     padding=padding, kernel_initializer=initializer)

# fc_caps -> DePrimaryCaps -> conv/deconv
def DePrimaryCaps(inputs, activation,
                  num_outputs,
                  output_feat_size, # 4 x 4 feature map size
                  re_fc_dim = [],
                  #kernel_size=4, 
                  #strides=2,
                  #padding='same',
                  ):
    '''
    the second version of deprimarycaps layer, which change the fc_cap layer into feature maps
    which can be processed by convolutional or deconvolutional layers. 
    PrimaryCaps layer did two thing: 1. conv 2. reshape to capsule
    DePrimaryCaps will do two thing: 1. for each capsule, connected with full connection layer, 
                                     2. reshape fc layers for output_feat_size feature maps and 
                                        concate the all reshaped feature maps as the output 
    Args:
        input: [batch_size, num_total_caps] + in_caps_shape
        activation: [batch_size, num_total_caps] # activation is not used in this layer
        num_outputs: total outputting feature map channels
        output_feat_size: the feature map size of the outputting of deprimarycaps: output_feat_size x output_feat_size
        re_fc_dim: a list. the fc layers setting between the fc_caps layer and the output layers. 
                   set [] means no extra fc layers
                   set [1000, 1000] mean two extra layers with 1000 neurons for each 
    Returns:
        CNN layer: [batch_size, H, W, C] # C = int(num_outputs / inputs_num_capsules) * intputs_num_capsules
    
    '''
    in_pose_shape = inputs.get_shape().as_list()
    batch_size = in_pose_shape[0]
    inputs_num_capsules = in_pose_shape[1]
    each_cap_fc_filter_num = int(num_outputs / inputs_num_capsules)

    split_values = []
    for i in range(inputs_num_capsules):
        split_values.append(1)

    output = []
    split_capsules = tf.split(inputs, split_values, axis=1)
    for i in range(inputs_num_capsules):
        # with tf.variable_scope('capsule_fc_%d'%i):
        fc_z = tf.reshape(split_capsules[i], shape=[batch_size, -1])
        if len(re_fc_dim) == 0:
            fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=output_feat_size * output_feat_size * each_cap_fc_filter_num)
            feat_caps_each = tf.reshape(fc_z, shape=[batch_size, output_feat_size, output_feat_size, each_cap_fc_filter_num])
        else:
            for j in range(re_fc_dim):
                fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=re_fc_dim[j])
            fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=output_feat_size * output_feat_size * each_cap_fc_filter_num)
            feat_caps_each = tf.reshape(fc_z, shape=[batch_size, output_feat_size, output_feat_size, each_cap_fc_filter_num])
        if len(tf.shape(output).get_shape().as_list()) < 2:
            output = feat_caps_each
        else:
            output = tf.concat([output, feat_caps_each], axis=3)
    return output 

# we design the multi outputs version of deprimarycaps layer, which outputs multi sizes feature maps.
def DePrimaryCaps_multi(inputs, activation,
                  num_outputs,
                  output_feat_size, # [] # 4 x 4 feature map size
                  re_fc_dim = [],
                  ):
    '''
    the second version of deprimarycaps layer, which change the fc_cap layer into feature maps
    which can be processed by convolutional or deconvolutional layers. 
    PrimaryCaps layer did two thing: 1. conv 2. reshape to capsule
    DePrimaryCaps will do two thing: 1. for each capsule, connected with full connection layer, 
                                     2. reshape fc layers for output_feat_size feature maps and 
                                        concate the all reshaped feature maps as the output 
    Args:
        input:      [batch_size, num_total_caps] + in_caps_shape
        activation: [batch_size, num_total_caps] # activation is not used in this layer
        num_outputs:      a list []. outputting feature map channels for each feature map
        output_feat_size: a list []. the feature map size of the outputting of deprimarycaps: output_feat_size x output_feat_size
        re_fc_dim:        a list []. the fc layers setting between the fc_caps layer and the output layers. 
                   set [] means no extra fc layers
                   set [1000, 1000] mean two extra layers with 1000 neurons for each 
    Returns:
        CNN layer: [batch_size, H, W, C] # C = int(num_outputs / inputs_num_capsules) * intputs_num_capsules
    
    '''

    assert len(num_outputs) == len(output_feat_size) # the number of outputting featmaps should be equal to the number of outputing channel categories
    in_pose_shape = inputs.get_shape().as_list()
    batch_size = in_pose_shape[0]
    inputs_num_capsules = in_pose_shape[1]

    assert inputs_num_capsules >= len(num_outputs) # at least we should allocate one capsule for one size of feature map
    each_alo_num = int(inputs_num_capsules / len(num_outputs))
    caps_alo_num = []
    for i in range(len(num_outputs)-1):
        caps_alo_num.append(each_alo_num) 
    caps_alo_num.append(inputs_num_capsules- each_alo_num*(len(num_outputs)-1)) # for the last feat map, just allocate the left capsules
    sp_cap_t = tf.split(inputs, caps_alo_num, axis=1)
    feat_maps = []
    for f in range(len(num_outputs)):
        split_values = []
        for i in range(caps_alo_num[f]):
            split_values.append(1)
        each_cap_fc_filter_num = max(1, int(num_outputs[f] / caps_alo_num[f])) 
        # for each capsule, how many filters should be connected, thus, we think at least we should transform one capsule into one feature map. 
        output = []
        split_capsules = tf.split(sp_cap_t[f], split_values, axis=1)
        for i in range(caps_alo_num[f]):
            # with tf.variable_scope('capsule_fc_%d'%i):
            fc_z = tf.reshape(split_capsules[i], shape=[batch_size, -1])
            if len(re_fc_dim) == 0:
                fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=output_feat_size[f] * output_feat_size[f] * each_cap_fc_filter_num)
                feat_caps_each = tf.reshape(fc_z, shape=[batch_size, output_feat_size[f], output_feat_size[f], each_cap_fc_filter_num])
            else:
                for j in range(re_fc_dim):
                    fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=re_fc_dim[j])
                fc_z = tf.contrib.layers.fully_connected(fc_z, num_outputs=output_feat_size[f] * output_feat_size[f] * each_cap_fc_filter_num)
                feat_caps_each = tf.reshape(fc_z, shape=[batch_size, output_feat_size[f], output_feat_size[f], each_cap_fc_filter_num])
            if len(tf.shape(output).get_shape().as_list()) < 2:
                output = feat_caps_each
            else:
                output = tf.concat([output, feat_caps_each], axis=3)
        feat_maps.append(output)
    print(feat_maps[0])
    return feat_maps 













