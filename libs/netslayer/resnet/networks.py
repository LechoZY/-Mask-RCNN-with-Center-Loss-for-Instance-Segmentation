import time

from libs.netslayer.resnet.ops import *
from libs.netslayer.resnet.utils import *


def network(inputs, res_n=50, is_training=True, num_classes=None, reuse=False, global_pool=True):
    with tf.variable_scope("network", reuse=reuse):
        if res_n < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        ch = 32 # paper is 64
        x = conv(inputs, channels=ch, kernel=3, stride=1, scope='conv')

        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

        ########################################################################################################


        x = batch_norm(x, is_training, scope='batch_norm')
        x = relu(x)
        if global_pool:
            x = global_avg_pooling(x)
        if num_classes is not None:
            x = fully_conneted(x, units=num_classes, scope='logit')

        return x