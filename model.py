'''
4-stacked hourglass network
The code is from https://github.com/umich-vl/pose-ae-demo/ written by Alejandro Newell
'''

import tensorflow as tf

def cnv(inp, kernel_shape, scope_name, stride=[1,1,1,1], dorelu=True,
    weight_init_fn=tf.random_normal_initializer,
    bias_init_fn=tf.constant_initializer, bias_init_val=0.0, pad='SAME',):

    with tf.variable_scope(scope_name):
        std = 1 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
        std = std ** .5
        weights = tf.get_variable('weights', kernel_shape,
                                  initializer=weight_init_fn(stddev=std))
        biases = tf.get_variable('biases', [kernel_shape[-1]],
                                  initializer=bias_init_fn(bias_init_val))
        conv = tf.nn.conv2d(inp, weights, strides=stride, padding=pad) + biases
        # Add ReLU
        if dorelu: return tf.nn.relu(conv)
        else: return conv

def pool1(inp, name=None, kernel=[2,2], stride=[2,2]):
    # Initialize max-pooling layer (default 2x2 window, stride 2)
    kernel = [1] + kernel + [1]
    stride = [1] + stride + [1]
    return tf.nn.max_pool(inp, kernel, stride, 'SAME', name=name)

def hourglass1(inp, n, f, hg_id):
    # Upper branch
    nf = f # nf = no of feature map
    up1 = cnv(inp, [3, 3, f, f], '%d_%d_up1' % (hg_id, n))
    #print("the size of up1", up1) # shape = (4, 512, 64, 256)

    # Lower branch
    pool = pool1(inp, '%d_%d_pool' % (hg_id, n))
    #print("the size of pool1", pool1) # shape = (4, 256, 32, 256)
    low1 = cnv(pool, [3, 3, f, nf], '%d_%d_low1' % (hg_id, n))
    #print("the size of low1", low1) # shape = (4, 256, 32, 256) , shape=(4, 32, 4, 256)
    # Recursive hourglass
    if n > 1:
        low2 = hourglass1(low1, n - 1, nf, hg_id)
        #print("the size of low2", low2)
    else:
        low2 = cnv(low1, [3, 3, nf, nf], '%d_%d_low2' % (hg_id, n))
        #print("the size of low2", low2)
    low3 = cnv(low2, [3, 3, nf, f], '%d_%d_low3' % (hg_id, n))
    #print("the size of low3", low3)

    up_size = tf.shape(up1)[1:3]
    #print("the size of up_size", up_size) # shape=(2,)
    up2 = tf.image.resize_nearest_neighbor(low3, up_size)
    #print("the size of up2", up2) # shape=(4, 64, 8, 256)
    #print("the sum of up1 + up2", up1 + up2)
    return up1 + up2

def infer1(inp_img, num_output_channel):
    f = 256
    cnv1 = cnv(inp_img, [7, 7, 1, 64], 'cnv1', stride=[1,1,1,1])
    cnv2 = cnv(cnv1, [3, 3, 64, 128], 'cnv2')
    #pool1 = pool(cnv2, 'pool1')
    cnv2b = cnv(cnv2, [3, 3, 128, 128], 'cnv2b')
    cnv3 = cnv(cnv2b, [3, 3, 128, 128], 'cnv3')
    cnv4 = cnv(cnv3, [3, 3, 128, f], 'cnv4')

    inter = cnv4
    #print("the inter size is",inter) # shape = (4,512,64,256)

    preds = []
    feature_map = []
    for i in range(4):
        # Hourglass
        hg = hourglass1(inter, 6, f, i)

        # Final output
        cnv5 = cnv(hg, [3, 3, f, f], 'cnv5_%d' % i)
        #print("the size of cnv5", cnv5)
        cnv6 = cnv(cnv5, [1, 1, f, f], 'cnv6_%d' % i)
        feature_map += [cnv6]
        #print("the size of cnv6", cnv6)
        preds += [cnv(cnv6, [1, 1, f, num_output_channel], 'out_%d' % i, dorelu=False)]
        
        #print("the size of preds", preds)

        # Residual link across hourglasses
        if i < 3:
            inter = inter + cnv(cnv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False)\
            + cnv(preds[-1], [1, 1, num_output_channel, f], 'tmp_out_%d'%i, dorelu = False)
    #print(preds[-1].shape)
    return feature_map, preds

def pool2(inp, name=None, kernel=[2,2], stride=[2,2]):
    # Initialize max-pooling layer (default 2x2 window, stride 2)
    kernel = [1] + kernel + [1]
    stride = [1] + stride + [1]
    return tf.nn.max_pool(inp, kernel, stride, 'SAME', name=name)

def hourglass2(inp, n, f, hg_id):
    # Upper branch
    nf = f # nf = no of feature map
    up1 = cnv(inp, [3, 3, f, f], 'two_%d_%d_up1' % (hg_id, n))
    #print("the size of up1", up1) # shape = (4, 512, 64, 256)

    # Lower branch
    pool = pool2(inp, 'two_%d_%d_pool' % (hg_id, n))
    #print("the size of pool1", pool1) # shape = (4, 256, 32, 256)
    low1 = cnv(pool, [3, 3, f, nf], 'two_%d_%d_low1' % (hg_id, n))
    #print("the size of low1", low1) # shape = (4, 256, 32, 256) , shape=(4, 32, 4, 256)
    # Recursive hourglass
    if n > 1:
        low2 = hourglass2(low1, n - 1, nf, hg_id)
        #print("the size of low2", low2)
    else:
        low2 = cnv(low1, [3, 3, nf, nf], 'two_%d_%d_low2' % (hg_id, n))
        #print("the size of low2", low2)
    low3 = cnv(low2, [3, 3, nf, f], 'two_%d_%d_low3' % (hg_id, n))
    #print("the size of low3", low3)

    up_size = tf.shape(up1)[1:3]
    #print("the size of up_size", up_size) # shape=(2,)
    up2 = tf.image.resize_nearest_neighbor(low3, up_size)
    #print("the size of up2", up2) # shape=(4, 64, 8, 256)
    #print("the sum of up1 + up2", up1 + up2)
    return up1 + up2

def infer2(inp_img, num_output_channel):
    f = 256
    cnv1 = cnv(inp_img, [7, 7, 1, 64], 'two_cnv1', stride=[1,1,1,1])
    cnv2 = cnv(cnv1, [3, 3, 64, 128], 'two_cnv2')
    #pool1 = pool(cnv2, 'pool1')
    cnv2b = cnv(cnv2, [3, 3, 128, 128], 'two_cnv2b')
    cnv3 = cnv(cnv2b, [3, 3, 128, 128], 'two_cnv3')
    cnv4 = cnv(cnv3, [3, 3, 128, f], 'two_cnv4')

    inter = cnv4
    #print("the inter size is",inter) # shape = (4,512,64,256)

    preds = []
    feature_map = []
    for i in range(4):
        # Hourglass
        hg = hourglass2(inter, 6, f, i)

        # Final output
        cnv5 = cnv(hg, [3, 3, f, f], 'two_cnv5_%d' % i)
        #print("the size of cnv5", cnv5)
        cnv6 = cnv(cnv5, [1, 1, f, f], 'two_cnv6_%d' % i)
        feature_map += [cnv6]
        #print("the size of cnv6", cnv6)
        preds += [cnv(cnv6, [1, 1, f, num_output_channel], 'two_out_%d' % i, dorelu=False)]
        
        #print("the size of preds", preds)

        # Residual link across hourglasses
        if i < 3:
            inter = inter + cnv(cnv6, [1, 1, f, f], 'two_tmp_%d' % i, dorelu=False)\
            + cnv(preds[-1], [1, 1, num_output_channel, f], 'two_tmp_out_%d'%i, dorelu = False)
    #print(preds[-1].shape)
    return feature_map, preds

def pool3(inp, name=None, kernel=[2,2], stride=[2,2]):
    # Initialize max-pooling layer (default 2x2 window, stride 2)
    kernel = [1] + kernel + [1]
    stride = [1] + stride + [1]
    return tf.nn.max_pool(inp, kernel, stride, 'SAME', name=name)

def hourglass3(inp, n, f, hg_id):
    # Upper branch
    nf = f # nf = no of feature map
    up1 = cnv(inp, [3, 3, f, f], 'three_%d_%d_up1' % (hg_id, n))
    #print("the size of up1", up1) # shape = (4, 512, 64, 256)

    # Lower branch
    pool = pool3(inp, 'three_%d_%d_pool' % (hg_id, n))
    #print("the size of pool1", pool1) # shape = (4, 256, 32, 256)
    low1 = cnv(pool, [3, 3, f, nf], 'three_%d_%d_low1' % (hg_id, n))
    #print("the size of low1", low1) # shape = (4, 256, 32, 256) , shape=(4, 32, 4, 256)
    # Recursive hourglass
    if n > 1:
        low2 = hourglass3(low1, n - 1, nf, hg_id)
        #print("the size of low2", low2)
    else:
        low2 = cnv(low1, [3, 3, nf, nf], 'three_%d_%d_low2' % (hg_id, n))
        #print("the size of low2", low2)
    low3 = cnv(low2, [3, 3, nf, f], 'three_%d_%d_low3' % (hg_id, n))
    #print("the size of low3", low3)

    up_size = tf.shape(up1)[1:3]
    #print("the size of up_size", up_size) # shape=(2,)
    up2 = tf.image.resize_nearest_neighbor(low3, up_size)
    #print("the size of up2", up2) # shape=(4, 64, 8, 256)
    #print("the sum of up1 + up2", up1 + up2)
    return up1 + up2

def infer3(inp_img, num_output_channel):
    f = 256
    cnv1 = cnv(inp_img, [7, 7, 1, 64], 'three_cnv1', stride=[1,1,1,1])
    cnv2 = cnv(cnv1, [3, 3, 64, 128], 'three_cnv2')
    #pool1 = pool(cnv2, 'pool1')
    cnv2b = cnv(cnv2, [3, 3, 128, 128], 'three_cnv2b')
    cnv3 = cnv(cnv2b, [3, 3, 128, 128], 'three_cnv3')
    cnv4 = cnv(cnv3, [3, 3, 128, f], 'three_cnv4')

    inter = cnv4
    #print("the inter size is",inter) # shape = (4,512,64,256)

    preds = []
    feature_map = []
    for i in range(4):
        # Hourglass
        hg = hourglass3(inter, 6, f, i)

        # Final output
        cnv5 = cnv(hg, [3, 3, f, f], 'three_cnv5_%d' % i)
        #print("the size of cnv5", cnv5)
        cnv6 = cnv(cnv5, [1, 1, f, f], 'three_cnv6_%d' % i)
        feature_map += [cnv6]
        #print("the size of cnv6", cnv6)
        preds += [cnv(cnv6, [1, 1, f, num_output_channel], 'three_out_%d' % i, dorelu=False)]
        
        #print("the size of preds", preds)

        # Residual link across hourglasses
        if i < 3:
            inter = inter + cnv(cnv6, [1, 1, f, f], 'three_tmp_%d' % i, dorelu=False)\
            + cnv(preds[-1], [1, 1, num_output_channel, f], 'three_tmp_out_%d'%i, dorelu = False)
    #print(preds[-1].shape)
    return feature_map, preds

def multi_band_multi_stack(inp_img,num_output_channel):
    band0, band1 = tf.split(inp_img, num_or_size_splits=2, axis=1)
    full_band = inp_img
    two_band_concat = []
    all_band_concat = []
    final_preds = []
    
    band0_feature_output, band0_preds = infer1(band0, num_output_channel)
    print("band0_feature_output", band0_feature_output)
    
    band1_feature_output, band1_preds = infer2(band1, num_output_channel)
    print("band1_feature_output", band1_feature_output)
    
    band2_feature_output, band2_preds = infer3(inp_img, num_output_channel)
    print("band2_feature_output", band2_feature_output)
    
    for i in range(len(band0_feature_output)):
        band_concat = tf.concat([band0_feature_output[i], band1_feature_output[i]], axis = 1)
        print("band_concat", band_concat)
        two_band_concat.append(band_concat)
        
    print("two_band_concat", two_band_concat)
        
    for i in range(len(band2_feature_output)):
        all_concat = tf.concat([two_band_concat[i], band2_feature_output[i]], axis = 3)
        print("all_concat", all_concat)
        all_band_concat.append(all_concat)
        
    print("all_band_concat", all_band_concat)
        
    first_stack_final_preds = cnv(all_band_concat[0], [3, 3, 512, 256], 'first_stack_1_cnv5')
    first_stack_final_preds = cnv(first_stack_final_preds, [1, 1, 256, 64], 'first_stack_2_cnv5')
    first_stack_final_preds = cnv(first_stack_final_preds, [1, 1, 64, num_output_channel], 'first_stack_3_cnv5')
    final_preds.append(first_stack_final_preds)
    
    second_stack_final_preds = cnv(all_band_concat[1], [3, 3, 512, 256], 'second_stack_1_cnv5')
    second_stack_final_preds = cnv(second_stack_final_preds, [1, 1, 256, 64], 'second_stack_2_cnv5')
    second_stack_final_preds = cnv(second_stack_final_preds, [1, 1, 64, num_output_channel], 'second_stack_3_cnv5')
    final_preds.append(second_stack_final_preds)
    
    third_stack_final_preds = cnv(all_band_concat[2], [3, 3, 512, 256], 'third_stack_1_cnv5')
    third_stack_final_preds = cnv(third_stack_final_preds, [1, 1, 256, 64], 'third_stack_2_cnv5')
    third_stack_final_preds = cnv(third_stack_final_preds, [1, 1, 64, num_output_channel], 'third_stack_3_cnv5')
    final_preds.append(third_stack_final_preds)
    
    fourth_stack_final_preds = cnv(all_band_concat[3], [3, 3, 512, 256], 'fourth_stack_1_cnv5')
    fourth_stack_final_preds = cnv(fourth_stack_final_preds, [1, 1, 256, 64], 'fourth_stack_2_cnv5')
    fourth_stack_final_preds = cnv(fourth_stack_final_preds, [1, 1, 64, num_output_channel], 'fourth_stack_3_cnv5')
    final_preds.append(fourth_stack_final_preds)
    
    print("final_preds", final_preds)
    
    return final_preds

    



