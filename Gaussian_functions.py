# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:55:00 2024

@author: Admin
"""

import numpy as np
import tensorflow as tf

def Gaussian(positions, tfv_params, atom_num_list, pad_size_list, im_shape):

    pad_array_list = []
    slices_list = []
    slices = 0

    lattice_num = positions.shape[1]/np.sum(np.array(atom_num_list))

    atom_type_num = len(atom_num_list)

    add_pix = np.max(pad_size_list)

    for i in range(atom_type_num):
    
        pad_size = pad_size_list[i]
        x_pad = np.arange(2*pad_size+1)
        y_pad = np.arange(2*pad_size+1)
        
        X, Y = np.meshgrid(x_pad, y_pad)
        
        pad_array_list.append(np.column_stack([X.ravel().astype(int), Y.ravel().astype(int)]))

        slices += lattice_num*atom_num_list[i]
        slices_list.append(slices)
    
    tf_white_board = tf.zeros((im_shape[0] + 2*add_pix, im_shape[1] + 2*add_pix), dtype = tf.float32)

    for i in range(tfv_params.shape[1]):

        if 0 <= i < slices_list[0]:
            tf_pad = tf.convert_to_tensor(pad_array_list[0], dtype=tf.float32) 
            pad_size = pad_size_list[0]
        elif slices_list[0] <= i < slices_list[1]:
            tf_pad = tf.convert_to_tensor(pad_array_list[1], dtype=tf.float32) 
            pad_size = pad_size_list[1]
        elif slices_list[1] <= i < slices_list[2]:
            tf_pad = tf.convert_to_tensor(pad_array_list[2], dtype=tf.float32) 
            pad_size = pad_size_list[2]
        else:
            tf_pad = tf.convert_to_tensor(pad_array_list[3], dtype=tf.float32)
            pad_size = pad_size_list[3]

        tf_x = tf.cast(positions[0,i], tf.int32) + add_pix
        tf_y = tf.cast(positions[1,i], tf.int32) + add_pix
        tf_A = tf.cast(tfv_params[0,i], tf.float32)
        tf_sig = tf.cast(tfv_params[1,i], tf.float32)
            
        tf_r_square = (tf_pad[:,0] - pad_size)**2 + (tf_pad[:,1] - pad_size)**2
        tf_array_value = tf_A*tf.exp(-tf_r_square/(2*tf_sig**2))

        tf_X, tf_Y = tf.meshgrid(tf.range(tf_x - pad_size, tf_x + pad_size + 1), tf.range(tf_y - pad_size, tf_y + pad_size + 1))
        tf_flatX = tf.reshape(tf_X, -1)
        tf_flatY = tf.reshape(tf_Y, -1)
        
        indices = tf.stack([tf_flatY, tf_flatX], axis = -1)

        tf_white_board = tf.tensor_scatter_nd_add(tf_white_board, indices, tf_array_value)
        cliped_tf_white_board = tf_white_board[add_pix : tf_white_board.shape[0] - add_pix, add_pix : tf_white_board.shape[1] - add_pix]

    return  cliped_tf_white_board
    
def Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape):

    np_params = np.array(tfv_params)

    pad_array_list = []
    slices_list = []
    slices = 0

    lattice_num = positions.shape[1]/np.sum(np.array(atom_num_list))

    atom_type_num = len(atom_num_list)

    add_pix = np.max(pad_size_list)
    
    for i in range(atom_type_num):
    
        pad_size = pad_size_list[i]
        x_pad = np.arange(2*pad_size+1)
        y_pad = np.arange(2*pad_size+1)
        
        X, Y = np.meshgrid(x_pad, y_pad)
        
        pad_array_list.append(np.column_stack([X.ravel().astype(int), Y.ravel().astype(int)]))

        slices += lattice_num*atom_num_list[i]
        slices_list.append(slices)
    
    white_board = np.zeros((im_shape[0] + 2*add_pix, im_shape[1] + 2*add_pix))

    for i in range(positions.shape[1]):

        if 0 <= i < slices_list[0]:
            pad = pad_array_list[0]
            pad_size = pad_size_list[0]
        elif slices_list[0] <= i < slices_list[1]:
            pad = pad_array_list[1]
            pad_size = pad_size_list[1]
        elif slices_list[1] <= i < slices_list[2]:
            pad = pad_array_list[2]
            pad_size = pad_size_list[2]
        else:
            pad = pad_array_list[3]
            pad_size = pad_size_list[3]
        
        x_cor = int(positions[0,i]) + add_pix
        y_cor = int(positions[1,i]) + add_pix
        A = np_params[0,i]
        sig = np_params[1,i]
            
        r_square = (pad[:,0] - pad_size)**2 + (pad[:,1] - pad_size)**2
        array_value = A*np.exp(-r_square/(2*sig**2))
        pad_value = array_value.reshape(2*pad_size+1, 2*pad_size+1)

        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] = \
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] + pad_value

        cliped_white_board = white_board[add_pix : white_board.shape[0] - add_pix, add_pix : white_board.shape[1] - add_pix]

    return  cliped_white_board

def Gaussian_draw_init(posit_pix, params_init, pad_size, figsize):

    total_num = 0

    for i in range(len(posit_pix)):

        total_num += len(posit_pix[i])

    positions = np.zeros((2, total_num))
    params = np.zeros((2, total_num))

    accul = 0

    for i in range(len(posit_pix)):

        positions[:,accul : accul + len(posit_pix[i])] = np.transpose(posit_pix[i])
        params[:,accul : accul + len(posit_pix[i])] = np.tile(params_init[:,i,np.newaxis], (1,len(posit_pix[i])))

        accul += len(posit_pix[i]) 

    pad_size = pad_size
    x_pad = np.arange(2*pad_size+1)
    y_pad = np.arange(2*pad_size+1)
    
    X, Y = np.meshgrid(x_pad, y_pad)
    
    pad = np.column_stack([X.ravel().astype(int), Y.ravel().astype(int)])

    add_pix = 2*pad_size
    
    white_board = np.zeros((figsize[0] + 2*add_pix, figsize[1] + 2*add_pix))

    for i in range(positions.shape[1]):
 
        x_cor = int(positions[0,i]) + add_pix 
        y_cor = int(positions[1,i]) + add_pix 
        A = params[0,i]
        sig = params[1,i]
            
        r_square = (pad[:,0] - pad_size)**2 + (pad[:,1] - pad_size)**2
        array_value = A*np.exp(-r_square/(2*sig**2))
        pad_value = array_value.reshape(2*pad_size+1, 2*pad_size+1)
        
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] = \
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] + pad_value

        white_board_clip = white_board[add_pix : white_board.shape[0] -add_pix, add_pix: white_board.shape[1] -add_pix]

    return  white_board_clip

def Gaussian_draw_high_resol(wa_pix_posit, wa_params, figsize, first_position: list, resol_amp, pad_size):

    total_num = 0

    for i in range(len(wa_pix_posit)):

        total_num += len(wa_pix_posit[i])

    positions = np.zeros((2, total_num))
    params = np.zeros((2, total_num))

    accul = 0

    for i in range(len(wa_pix_posit)):

        positions[:,accul : accul + len(wa_pix_posit[i])] = np.transpose(wa_pix_posit[i])
        params[:,accul : accul + len(wa_params[i])] = np.transpose(wa_params[i])

        accul += len(wa_pix_posit[i]) 

    positions = positions - np.tile(positions[:,0,np.newaxis], (1,total_num)) + np.tile(np.array(first_position)[:,np.newaxis], (1,total_num)) 
    params[1,:] = params[1,:]*resol_amp

    highresol_positions = positions*resol_amp
    highresol_positions = np.rint(highresol_positions).astype(int)

    pad_size = pad_size*resol_amp
    
    x_pad = np.arange(2*pad_size+1)
    y_pad = np.arange(2*pad_size+1)
    
    X, Y = np.meshgrid(x_pad, y_pad)
    
    pad = np.column_stack([X.ravel().astype(int), Y.ravel().astype(int)])

    add_pix = 2*pad_size
    
    white_board = np.zeros((figsize[0]*resol_amp + 2*add_pix, figsize[1]*resol_amp + 2*add_pix))

    for i in range(highresol_positions.shape[1]):
 
        x_cor = int(highresol_positions[0,i]) + add_pix 
        y_cor = int(highresol_positions[1,i]) + add_pix 
        A = params[0,i]
        sig = params[1,i]
            
        r_square = (pad[:,0] - pad_size)**2 + (pad[:,1] - pad_size)**2
        array_value = A*np.exp(-r_square/(2*sig**2))
        pad_value = array_value.reshape(2*pad_size+1, 2*pad_size+1)
        
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] = \
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] + pad_value

        white_board_clip = white_board[add_pix : white_board.shape[0] -add_pix, add_pix: white_board.shape[1] -add_pix]

    return  white_board_clip, positions
