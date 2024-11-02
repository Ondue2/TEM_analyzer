# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:48:51 2024

@author: Admin
"""

import tensorflow as tf
import tools
import Gaussian_functions as gf
import loss_functions as lf
import numpy as np
import copy
import matplotlib.pyplot as plt

def lattice_optimization(num_epoch, print_num, num_amp, num_posit, learning_rate, lattices_update, positions_update, positions, im_analyzed, tf_im_analyzed,
                         loss_array, tf_amplitude_init, tf_width_init, x_l, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, atom_num_list, pad_size_list, 
                        x_update_step, y_update_step, row, col, len_pix, lattice_num, base_dir, lattice_atom_positions_file_name, 
                        params_init_file_name, every_positions_init_file_name, every_params_init_file_name, loss_file_name):
    
    im_shape = im_analyzed.shape

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    tf_amplitude = tools.tf_parameters_gen(tf_amplitude_init, lattice_num, atom_num_list)
    tf_width = tools.tf_parameters_gen(tf_width_init, lattice_num, atom_num_list) 
    tfv_params = tf.stack([tf_amplitude, tf_width])
    
    np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
    loss = lf.compute_loss_np(im_analyzed, np_simul_im)
    loss_array.append(loss)
    
    for epoch in range(num_epoch):  
        
    
        # Amplitude optimization
    
        for step in range(num_amp):
            with tf.GradientTape() as tape:
    
                tf_amplitude = tools.tf_parameters_gen(tf_amplitude_init, lattice_num, atom_num_list)
                tf_width = tools.tf_parameters_gen(tf_width_init, lattice_num, atom_num_list)
    
                tfv_params = tf.stack([tf_amplitude, tf_width])
                
                np.save(base_dir + every_params_init_file_name, np.array(tfv_params))
        
                simulated_image = gf.Gaussian(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
        
                loss = lf.compute_loss_tf(tf_im_analyzed, simulated_image)
                loss_array.append(loss)
                loss_recent = loss
        
                grads = tape.gradient(loss, [tf_amplitude_init, tf_width_init])
        
                optimizer.apply_gradients(zip(grads, [tf_amplitude_init, tf_width_init]))
    
                np.save(base_dir + params_init_file_name, np.array([tf_amplitude_init, tf_width_init]))
    
        np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape)  
        loss = lf.compute_loss_np(im_analyzed, np_simul_im)
        loss_recent = loss   
        
        # Positions optimization
        
        for step in range(num_posit):
            
    
            # lattice constant x
            
            positions = tools.positions_init_gen(x_l+x_update_step, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_p = loss_recent - current_loss
    
            positions = tools.positions_init_gen(x_l-x_update_step, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_n = loss_recent - current_loss
            
            prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
            prob = prob/np.sum(prob)
        
            step = np.random.choice([0, x_update_step, -x_update_step], p = prob)
            x_l_step = step
            
    
            # lattice constant y
            
            positions = tools.positions_init_gen(x_l, y_l+y_update_step, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_p = loss_recent - current_loss
    
            positions = tools.positions_init_gen(x_l, y_l-y_update_step, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_n = loss_recent - current_loss
            
            prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
            prob = prob/np.sum(prob)
            
            step = np.random.choice([0, y_update_step, -y_update_step], p = prob)
            y_l_step = step
    
    
            # x offset
    
            positions = tools.positions_init_gen(x_l, y_l, x_off_l+1, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_p = loss_recent - current_loss
    
            positions = tools.positions_init_gen(x_l, y_l, x_off_l-1, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_n = loss_recent - current_loss
            
            prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
            prob = prob/np.sum(prob)
        
            step = np.random.choice([0, 1, -1], p = prob)
            x_off_l_step = step
    
    
            # y offset
    
            positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l+1, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_p = loss_recent - current_loss
    
            positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l-1, sliding_l, posit_pix_l, row, col, len_pix)
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
            current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
            adv_n = loss_recent - current_loss
            
            prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
            prob = prob/np.sum(prob)
        
            step = np.random.choice([0, 1, -1], p = prob)
            y_off_l_step = step
    
    
            # Sliding
    
            sliding_l_step = [0]*len(sliding_l)
            
            for i in range(len(sliding_l)):
    
                updated_sliding = sliding_l.copy()
                updated_sliding[i] = sliding_l[i] + 1
    
                positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l, updated_sliding, posit_pix_l, row, col, len_pix)
    
                np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape)  
                current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
                adv_p = loss_recent - current_loss
    
                updated_sliding = sliding_l.copy()
                updated_sliding[i] = sliding_l[i] - 1
    
                positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l, updated_sliding, posit_pix_l, row, col, len_pix)
    
                np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape)  
                current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
                adv_n = loss_recent - current_loss
    
                prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
                prob = prob/np.sum(prob)
            
                step = np.random.choice([0, 1, -1], p = prob)
                sliding_l_step[i] = step
                
    
            # Atomic positions
            
            posit_pix_l_step = []
            
            for i in range(len(posit_pix_l)):
    
                posit_pix_l_step.append(np.zeros((posit_pix_l[i].shape)))
      
            for i in range(len(posit_pix_l)):
                for j in range(len(posit_pix_l[i])):
                    for k in range(2):
    
                        update_posit_pixel_l = copy.deepcopy(posit_pix_l)
                        update_posit_pixel_l[i][j, k] = update_posit_pixel_l[i][j, k] + 1
        
                        positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l, sliding_l, update_posit_pixel_l, row, col, len_pix)
            
                        np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                        current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
                        adv_p = loss_recent - current_loss
    
                        update_posit_pixel_l = copy.deepcopy(posit_pix_l)
                        update_posit_pixel_l[i][j, k] = update_posit_pixel_l[i][j, k] - 1
        
                        positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l, sliding_l, update_posit_pixel_l, row, col, len_pix)
            
                        np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                        current_loss = lf.compute_loss_np(im_analyzed, np_simul_im)
                        adv_n = loss_recent - current_loss
    
                        prob = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
                        prob = prob/np.sum(prob)
                    
                        step = np.random.choice([0, 1, -1], p = prob)
                                
                        posit_pix_l_step[i][j,k] = step
    
            if lattices_update == True:
                x_l = x_l + x_l_step
                y_l = y_l + y_l_step
                x_off_l = x_off_l + x_off_l_step
                y_off_l = y_off_l + y_off_l_step
                sliding_l = [sliding_ + sliding_step_ for sliding_, sliding_step_ in zip(sliding_l, sliding_l_step)]
    
            if positions_update == True:
                posit_pix_l = [posit_pix_l_ + posit_pix_l_step_ for posit_pix_l_, posit_pix_l_step_ in zip(posit_pix_l, posit_pix_l_step)]
    
            positions = tools.positions_init_gen(x_l, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    
            lattice_atom_positions = np.array([x_l, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l], dtype = object)
            
            np.save(base_dir + every_positions_init_file_name, positions)
            np.save(base_dir + lattice_atom_positions_file_name, lattice_atom_positions)
            
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
        
            loss_recent  = lf.compute_loss_np(im_analyzed, np_simul_im)
    
            loss = loss_recent
            loss_array.append(loss)
            np.save(base_dir + loss_file_name, np.array(loss_array))
        
        if epoch % print_num == 0:
            
            print(f'epoch {epoch} current_loss_data {loss}')
            
    plt.plot(np.arange(len(loss_array)), loss_array)
    
def free_atom_optimization(num_epoch, print_num, num_amp, num_posit, learning_rate, gamma, reg_params, reg_posits, positions, tfv_params,
                          positions_init, tfv_params_init, atom_num_list, pad_size_list, im_analyzed, tf_im_analyzed,
                          base_dir, positions_file_name, params_file_name, loss_file_name, loss_array):

    im_shape = im_analyzed.shape
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
    loss = lf.compute_loss_np(im_analyzed, np_simul_im)
    loss_array.append(loss)
    
    for epoch in range(num_epoch):  
    
        for step in range(num_amp):
            with tf.GradientTape() as tape:
        
                simulated_image = gf.Gaussian(positions, tfv_params, atom_num_list, pad_size_list, im_shape)
        
                loss = lf.compute_loss_tf(tf_im_analyzed, simulated_image)
                loss_array.append(loss)
    
                loss_regul_p = lf.regulation_params(tfv_params, tfv_params_init, reg_params)
    
                loss = loss + loss_regul_p
        
                grads = tape.gradient(loss, tfv_params)
        
                optimizer.apply_gradients([(grads, tfv_params)])
    
                np.save(base_dir + params_file_name, np.array(tfv_params))
     
        np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape)
        loss = lf.compute_loss_np(im_analyzed, np_simul_im)
        loss_regul = lf.regulation_positions(positions, positions_init, reg_posits)
        loss_recent = loss + loss_regul   
    
        for step in range(num_posit):
    
            step_array = np.zeros((positions.shape))
            
            for pos in range(positions.shape[1]):
    
                    updated_positions = copy.deepcopy(positions)
                    updated_positions[0,pos] = positions[0,pos] + 1
    
                    np_simul_im = gf.Gaussian_position(updated_positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                    current_loss_data = lf.compute_loss_np(im_analyzed, np_simul_im)
                    current_loss_regul = lf.regulation_positions(updated_positions, positions_init, reg_posits)
                    current_loss_tot = current_loss_data + current_loss_regul
                    adv_p = loss_recent - current_loss_tot
    
                    updated_positions = copy.deepcopy(positions)
                    updated_positions[0,pos] = positions[0,pos] - 1
    
                    np_simul_im = gf.Gaussian_position(updated_positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                    current_loss_data = lf.compute_loss_np(im_analyzed, np_simul_im)
                    current_loss_regul = lf.regulation_positions(updated_positions, positions_init, reg_posits)
                    current_loss_tot = current_loss_data + current_loss_regul
                    adv_n = loss_recent - current_loss_tot
    
                    p_fav = (adv_p - adv_n)/(np.abs(adv_p) + np.abs(adv_n) + 1e-10)
                    prob_v = np.array([1 - np.abs(p_fav), np.abs(p_fav)*(1+p_fav)/2, np.abs(p_fav)*(1-p_fav)/2])
                    prob_b = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
                    prob = prob_v*gamma + prob_b*(1-gamma)
                    prob = prob/np.sum(prob)
                
                    step = np.random.choice([0, 1, -1], p = prob)
                    step_array[0, pos] = step
                
                    updated_positions = copy.deepcopy(positions)
                    updated_positions[1,pos] = positions[1,pos] + 1
    
                    np_simul_im = gf.Gaussian_position(updated_positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                    current_loss_data = lf.compute_loss_np(im_analyzed, np_simul_im)
                    current_loss_regul = lf.regulation_positions(updated_positions, positions_init, reg_posits)
                    current_loss_tot = current_loss_data + current_loss_regul
                    adv_p = loss_recent - current_loss_tot
    
                    updated_positions = copy.deepcopy(positions)
                    updated_positions[1,pos] = positions[1,pos] - 1
    
                    np_simul_im = gf.Gaussian_position(updated_positions, tfv_params, atom_num_list, pad_size_list, im_shape) 
                    current_loss_data = lf.compute_loss_np(im_analyzed, np_simul_im)
                    current_loss_regul = lf.regulation_positions(updated_positions, positions_init, reg_posits)
                    current_loss_tot = current_loss_data + current_loss_regul
                    adv_n = loss_recent - current_loss_tot
    
                    p_fav = (adv_p - adv_n)/(np.abs(adv_p) + np.abs(adv_n) + 1e-10)
                    prob_v = np.array([1 - np.abs(p_fav), np.abs(p_fav)*(1+p_fav)/2, np.abs(p_fav)*(1-p_fav)/2])
                    prob_b = np.where([0, adv_p, adv_n] == np.max([0, adv_p, adv_n]), 1, 0)
                    prob = prob_v*gamma + prob_b*(1-gamma)
                    prob = prob/np.sum(prob)
                
                    step = np.random.choice([0, 1, -1], p = prob)
                    step_array[1, pos] = step
    
            positions = positions + step_array
            np.save(base_dir + positions_file_name, np.array(positions))
    
            np_simul_im = gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape)
            loss_data = lf.compute_loss_np(im_analyzed, np_simul_im)
            loss_regul = lf.regulation_positions(positions, positions_init, reg_posits)
            loss_tot = loss_data + loss_regul
            loss_recent = loss_tot  
            
            loss = loss_data 
            loss_array.append(loss)
            np.save(base_dir + loss_file_name, np.array(loss_array))
        
        if epoch % 1 == 0:
            
            print(f'epoch {epoch} current_loss_data {loss} parameter regulation {loss_regul_p} position regulation {loss_regul}')
            
    plt.plot(np.arange(len(loss_array)), loss_array)