# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:24:15 2024

@author: Admin
"""

import numpy as np
import math
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import tensorflow as tf
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import copy
import Gaussian_functions as gf

# This is the class for the image rotation and clip

class im_process():
    def __init__(self, image):
        self.image = image
        self.shape = image.shape
        self.im_center = ((self.shape[0]-1)//2, (self.shape[1]-1)//2)
        self.background = np.min(image)

        self.full_len = math.ceil(np.sqrt(self.shape[0]**2 + self.shape[1]**2))
        self.new_im_cen = ((self.full_len)//2, (self.full_len)//2)

    def rotation(self, theta):
       
        the_r = theta*np.pi/180
        new_image = np.full((self.full_len+1, self.full_len+1), self.background)
        self.ref_image = np.zeros((self.full_len+1, self.full_len+1))
    
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                i_cen = i - self.im_center[0]
                j_cen = j - self.im_center[1]
    
                i_r = np.cos(the_r)*i_cen - np.sin(the_r)*j_cen 
                j_r = np.sin(the_r)*i_cen + np.cos(the_r)*j_cen
    
                new_i = int(i_r + self.new_im_cen[0])
                new_j = int(j_r + self.new_im_cen[1])
    
                new_image[new_i, new_j] = self.image[i, j]
                self.new_im = new_image

                self.ref_image[new_i, new_j] = 1
                
        return new_image

    def rect(self, u, b, l, r, linewidth):

        x = l
        y = u
        rect = Rectangle((x,y), r-l, b-u, linewidth = linewidth, edgecolor = 'red', facecolor = 'none')
        self.u = u
        self.b = b
        self.l = l
        self.r = r

        return rect

    def clip(self):

        cliped_image = self.new_im[self.u: self.b, self.l:self.r].copy()
        cliped_ref_image = self.ref_image[self.u: self.b, self.l:self.r].copy()
        defects = np.where(cliped_ref_image==0)

        corrected_cliped_image = cliped_image.copy()

        for i, j in zip(defects[0], defects[1]):
                if 0 < i < cliped_image.shape[0]-1 and 0 < j < cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j+1], cliped_image[i+1, j], cliped_image[i+1, j-1],
                                                           cliped_image[i, j+1], cliped_image[i, j-1],
                                                           cliped_image[i-1, j+1], cliped_image[i-1, j], cliped_image[i-1, j-1]]))
                    
                elif 0 < i < cliped_image.shape[0]-1 and j == 0:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j+1], cliped_image[i+1, j],
                                                           cliped_image[i, j+1], 
                                                           cliped_image[i-1, j+1], cliped_image[i-1, j]]))

        
                elif 0 < i < cliped_image.shape[0]-1 and j == cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j], cliped_image[i+1, j-1],
                                                           cliped_image[i, j-1],
                                                           cliped_image[i-1, j], cliped_image[i-1, j-1]]))

                elif i == 0 and 0 < j < cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j+1], cliped_image[i+1, j], cliped_image[i+1, j-1],
                                                           cliped_image[i, j+1], cliped_image[i, j-1],
                                                           ]))

                elif i == cliped_image.shape[0]-1 and 0 < j < cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([
                                                           cliped_image[i, j+1], cliped_image[i, j-1],
                                                           cliped_image[i-1, j+1], cliped_image[i-1, j], cliped_image[i-1, j-1]]))

                elif i == 0 and j == 0:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j+1], cliped_image[i+1, j],
                                                           cliped_image[i, j+1]
                                                           ]))

                elif i == 0 and j == cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([cliped_image[i+1, j], cliped_image[i+1, j-1],
                                                           cliped_image[i, j-1],
                                                            ]))

                elif i == cliped_image.shape[0]-1 and j == 0:
                    corrected_cliped_image[i, j] = int(np.mean([
                                                           cliped_image[i, j+1], 
                                                           cliped_image[i-1, j+1], cliped_image[i-1, j]]))

                elif i == cliped_image.shape[0]-1 and j == cliped_image.shape[1]-1:
                    corrected_cliped_image[i, j] = int(np.mean([
                                                           cliped_image[i, j-1],
                                                           cliped_image[i-1, j], cliped_image[i-1, j-1]]))
                                                       
        return cliped_image, corrected_cliped_image
    
    
# This is the class for lattice points generation

class lattice_position():
    def __init__(self, x, y, x_off, y_off, len_pix, row, col, sliding):
        
        self.x = x
        self.y = y
        self.x_off = x_off
        self.y_off = y_off
        self.len_pix = len_pix
        self.row = row
        self.col = col
        self.sliding = sliding
        
    def lattices(self):

        x = np.arange(self.col)*self.x/self.len_pix + self.x_off
        y = np.arange(self.row)*self.y/self.len_pix + self.y_off

        X, Y = np.meshgrid(x.astype(int), y.astype(int))

        lattices = np.column_stack([X.ravel(), Y.ravel()])

        for i in range(self.row - 1):

            lattices[(i+1)*self.col : (i+2)*self.col,0] = lattices[(i+1)*self.col : (i+2)*self.col,0] + self.sliding[i]
        
        return lattices

    def boxes(self, lattices):

        patches = [Rectangle((i, j), self.x/self.len_pix, self.y/self.len_pix) for i, j in lattices]
        collection = PatchCollection(patches, edgecolor='black', facecolor = 'none')
        
        return collection

    def overlapping(self, image, lattices):

        image_array = np.zeros((self.row*self.col, int(self.y/self.len_pix), int(self.x/self.len_pix)))

        m = 0

        for i in range(self.row):
            for j in range(self.col):
                
                image_array[m] = image[lattices[m, 1] : lattices[m, 1] + int(self.y/self.len_pix), 
                lattices[m, 0] : lattices[m, 0] + int(self.x/self.len_pix)]

                m += 1

        sum_image = np.sum(image_array, axis = 0)      
                        
        return image_array, sum_image/len(image_array)

    def recreating(self, sum_image, row_r, col_r):

        re_image = np.full((int(row_r*self.y/self.len_pix), int(col_r*self.x/self.len_pix)), np.min(sum_image))

        for i in range(row_r):
            for j in range(col_r):
                
                re_image[i*int(self.y/self.len_pix) : (i+1)*int(self.y/self.len_pix), 
                j*int(self.x/self.len_pix) : (j+1)*int(self.x/self.len_pix)] = sum_image

        return re_image
        
    
def position_gen(lattices, posit_pix):

    
    lattice_num = len(lattices)
    atom_type_num = len(posit_pix)

    total_num = 0
    atom_num_list = []

    for i in range(atom_type_num):

        total_num += lattice_num*len(posit_pix[i])
        atom_num_list.append(len(posit_pix[i]))
    
    positions_x = np.zeros((total_num))
    positions_y = np.zeros((total_num))

    accul = 0
          
    for i in range(atom_type_num):
        for j in range(lattice_num):
            
            positions_x[j*atom_num_list[i] + accul : (j+1)*atom_num_list[i] + accul] = posit_pix[i][:,0] + lattices[j,0]
            positions_y[j*atom_num_list[i] + accul : (j+1)*atom_num_list[i] + accul] = posit_pix[i][:,1] + lattices[j,1]

        accul += lattice_num*atom_num_list[i]

    positions = np.array([positions_x, positions_y])                     

    return positions
    

def unpack_atom_type(packed_array, lattice_num, atom_num_list):

    atom_type_num = len(atom_num_list)
    unpacked_array = []
    
    accul = 0

    for i in range(atom_type_num):

        unpacked_array.append(packed_array[:, accul : accul + lattice_num*atom_num_list[i]])
        accul += lattice_num*atom_num_list[i]

    return unpacked_array

def image_preprocess(TEM_array, rotate_angle, normalized_maximum_intensity, low_intensity_filter, up, down, left, right, line_width):

    image = im_process(TEM_array)

    rotated_im = image.rotation(rotate_angle)  
    rect = image.rect(up, down, left, right, 1)
    cliped_image, corrected_cliped_image = image.clip()
    
    im_analyzed = corrected_cliped_image
    im_analyzed = im_analyzed - np.min(im_analyzed)
    im_analyzed = np.where(im_analyzed < low_intensity_filter*np.max(im_analyzed), 0, im_analyzed - low_intensity_filter*np.max(im_analyzed))
    
    im_analyzed = normalized_maximum_intensity*im_analyzed/np.max(im_analyzed)
    tf_im_analyzed = tf.convert_to_tensor(im_analyzed, dtype=tf.float32)       # For the futrue use with tensorflow
    
    plt.figure()
    plt.title("Rotated TEM image")
    plt.imshow(rotated_im , cmap = "grey")
    plt.gca().add_patch(rect)

    plt.show()
    
    plt.figure()
    plt.title("Processed TEM image")
    plt.imshow(im_analyzed, cmap = "grey")
    plt.show()

    return im_analyzed, tf_im_analyzed

def lattice_gen_check(x, y, x_off, y_off, len_pix, row, col, sliding, im_analyzed):

    lattice_num = row*col
    lat = lattice_position(x, y, x_off, y_off, len_pix, row, col, sliding)  #put lattice information to the class
    lattices = lat.lattices()
    
    image_array, sum_image = lat.overlapping(im_analyzed, lattices)
    
    plt.figure()
    plt.title("Lattice information looks vaild?")
    plt.imshow(im_analyzed, cmap = "grey")
    
    collection = lat.boxes(lattices)
    
    plt.gca().add_collection(collection)
    
    fig, ax = plt.subplots(row, col)
    fig.suptitle("TEM image with the unit cell grid")
    
    m = 0
    
    for i in range(row):
        for j in range(col):
            ax[i][j].imshow(image_array[m], cmap = "grey")
            ax[i][j].axis('off')
            m += 1
    
    def plot_interact(horizon, vertical):
    
        horizon = int(horizon)
        vertical = int(vertical)
        fig, ax = plt.subplots(1, 3, figsize = (10, 5))
        fig.suptitle("TEM image of the average unit cell and intensity investigators")
        ax[0].imshow(sum_image, cmap = "grey")
        ax[0].axhline(y = horizon, color = 'red', linewidth = 1)
        ax[0].axvline(x = vertical, color = 'blue', linewidth = 1)
    
    
        ax[1].plot(sum_image[horizon,:])
        ax[1].set_title("horizontal")
        ax[1].set_xlim(0, sum_image.shape[1]-1)
        ax[2].plot(sum_image[:,vertical])
        ax[2].set_title("vertical")
        ax[2].set_xlim(0, sum_image.shape[0]-1)
    
    interact(plot_interact, horizon=widgets.FloatSlider(min=0, max=sum_image.shape[0]-1, step=1, value=0), 
             vertical=widgets.FloatSlider(min=0, max=sum_image.shape[1]-1, step=1, value=0));
    
    return sum_image, lattices, lattice_num

def atom_positions_iu_gen_check(Atom_positions_dic, sum_image, x, y, len_pix):

    atom_type_num = len(Atom_positions_dic)
    posit_pix = copy.deepcopy(list(Atom_positions_dic.values()))
    atom_num_list = [len(posit_pix[i]) for i in range(atom_type_num)]
    
    plt.figure()
    plt.title("Atomic positions look vaild?", pad = 15)
    plt.imshow(sum_image, cmap = "gray")
    
    for atom in range(len(posit_pix)):
    
        posit_pix[atom] = np.array(posit_pix[atom])
    
        posit_pix[atom][:,0] = posit_pix[atom][:,0]*int(x/len_pix)
        posit_pix[atom][:,1] = posit_pix[atom][:,1]*int(y/len_pix)
    
        posit_pix[atom] = posit_pix[atom].astype(int)
    
        plt.scatter(posit_pix[atom][:,0], posit_pix[atom][:,1])
        
    return posit_pix, atom_num_list

def positions_init_gen(x, y, x_off, y_off, sliding, posit_pix, row, col, len_pix):

        x = np.arange(col)*x/len_pix + x_off
        y = np.arange(row)*y/len_pix + y_off

        X, Y = np.meshgrid(np.rint(x).astype(int), np.rint(y).astype(int))

        lattices = np.column_stack([X.ravel(), Y.ravel()])     

        return position_gen(lattices, posit_pix)
    
def tf_parameters_gen(tf_value, lattice_num, atom_num_list):

    accul = 0
    
    param_pieces = []
    
    for i in range(len(atom_num_list)):
    
        param_piece = tf.fill([atom_num_list[i]*lattice_num], tf_value[i])
        param_pieces.append(param_piece)
        accul += atom_num_list[i]*lattice_num
    
    param = tf.concat(param_pieces, axis=0)

    return param

def params_init_to_draw(posit_pix, amplitude, width):

    atom_type_num = len(posit_pix)
    atom_num_list = []

    total_num = 0

    for i in range(atom_type_num):

        atom_num_list.append(len(posit_pix[i]))
        total_num += len(posit_pix[i])

    positions = position_gen(np.array([[0, 0]]), posit_pix)

    amplitude_np = np.zeros(total_num)
    width_np = np.zeros(total_num)

    accul = 0

    for i in range(atom_type_num):

        amplitude_np[accul : accul + atom_num_list[i]] = np.full(atom_num_list[i], amplitude[i])
        width_np[accul : accul + atom_num_list[i]] = np.full(atom_num_list[i], width[i])

        accul += atom_num_list[i]

    params = np.array([amplitude_np, width_np])

    return positions, params

def lattice_optimization_init(x, y, x_off, y_off, sliding, posit_pix, amplitude_pri, width_pri, row, col, len_pix):

    x_l = x 
    y_l = y
    x_off_l = x_off 
    y_off_l = y_off
    sliding_l = sliding.copy()
    posit_pix_l = copy.deepcopy(posit_pix)
    
    tf_amplitude_init = tf.Variable(amplitude_pri, dtype = tf.float32)
    tf_width_init =  tf.Variable(width_pri, dtype = tf.float32) 

    positions = positions_init_gen(x, y, x_off, y_off, sliding, posit_pix, row, col, len_pix)
    positions_pri = copy.deepcopy(positions)

    return x_l, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, tf_amplitude_init, tf_width_init, positions, positions_pri

def Lattice_optimization_results(base_dir, lattice_atom_positions_file_name, params_init_file_name, every_positions_init_file_name, 
                                 every_params_init_file_name, lattice_num, atom_num_list, row, col, len_pix, 
                                  im_analyzed, pad_size_list):

    im_shape = im_analyzed.shape
    
    lattice_atom_positions = list(np.load(base_dir + lattice_atom_positions_file_name + ".npy", allow_pickle=True))
    
    x_l = lattice_atom_positions[0]
    y_l = lattice_atom_positions[1]
    x_off_l = lattice_atom_positions[2]
    y_off_l = lattice_atom_positions[3]
    sliding_l = lattice_atom_positions[4]
    posit_pix_l = lattice_atom_positions[5]
    params = np.load(base_dir + params_init_file_name + ".npy")
    
    tfv_params = np.load(base_dir + every_params_init_file_name + ".npy")
    
    positions = positions_init_gen(x_l, y_l, x_off_l, y_off_l, sliding_l, posit_pix_l, row, col, len_pix)
    atom_resolved_positions = unpack_atom_type(positions, lattice_num, atom_num_list)
    
    plt.imshow(im_analyzed, cmap = "gray")
    plt.title("Positions with optimized lattice information")
    
    for atom in range(len(atom_num_list)):
        
        plt.scatter(atom_resolved_positions[atom][0,:], atom_resolved_positions[atom][1,:], s = 15)
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    ax[0].imshow(im_analyzed, cmap = "gray")
    ax[0].set_title("Experimental image")
    ax[1].imshow(gf.Gaussian_position(positions, tfv_params, atom_num_list, pad_size_list, im_shape), cmap = "gray")
    ax[1].set_title("Fitted image")
    
    amplitude_l = params[0]
    width_l = params[1]
    
    Lattice_optimization_parameters = (f'x_l: {x_l}, y_l: {y_l}, x_off_l: {x_off_l}, y_off_l: {y_off_l}\n \
    sliding_l: {sliding_l}\n \
    posit_pix_l: \n {posit_pix_l}\n \
    Gaussian amplitude: {amplitude_l}\n \
    Gaussian width: {width_l}')
    
    print(Lattice_optimization_parameters)
    print("\n\n Above parameters was saved in \"base_dir + params_lattice_optimization_recent.txt\"\n\n")
    
    with open(base_dir + "Atoms_params.txt", "w") as file:
        file.write(Lattice_optimization_parameters)

    return positions, tfv_params, posit_pix_l, params, atom_resolved_positions

def free_atom_opmization_results(base_dir, positions_file_name, params_file_name, lattice_num, atom_num_list, pad_size_list, im_analyzed):

    im_shape = im_analyzed.shape
    
    positions = np.load(base_dir + positions_file_name + ".npy")
    params = np.load(base_dir + params_file_name + ".npy")
    tfv_params = tf.Variable(params, dtype = tf.float32)
    
    atom_resolved_positions = unpack_atom_type(positions, lattice_num, atom_num_list)
    
    plt.imshow(im_analyzed, cmap = "gray")
    plt.title("Positions with optimized every single atom information")
    
    for atom in range(len(atom_num_list)):
        
        plt.scatter(atom_resolved_positions[atom][0,:], atom_resolved_positions[atom][1,:], s = 10)
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    ax[0].imshow(im_analyzed, cmap = "gray")
    ax[0].set_title("Experimental image")
    ax[1].imshow(gf.Gaussian_position(positions, params, atom_num_list, pad_size_list, im_shape), cmap = "gray")
    ax[1].set_title("Fitted image")

    return positions, params, tfv_params, atom_resolved_positions

def boundary_cut(atom_resolved_positions, atom_resolved_params, row, col, cut_row = 0, cut_col = 0):

    lattice_num = row*col

    cut_atom_positions = []
    cut_atom_params = []

    for atom in range(len(atom_resolved_positions)):
        
        total_num = int(atom_resolved_positions[atom].shape[1])
        
        atoms_in_lattice = int(total_num/lattice_num)
    
        cut_row_indices_top = np.arange(0 , cut_row*atoms_in_lattice*col) 
        cut_row_indices_botom = np.arange(total_num - cut_row*atoms_in_lattice*col , total_num)
        cut_row_indices = np.concatenate([cut_row_indices_top, cut_row_indices_botom])

        posit_row_cut = np.delete(atom_resolved_positions[atom], cut_row_indices, axis = 1)
        par_row_cut = np.delete(atom_resolved_params[atom], cut_row_indices, axis = 1)

        cut_col_indices_list = []

        atoms_in_row = col*atoms_in_lattice

        for i in range(row - 2*cut_row):
    
            cut_col_indices_list.append(np.arange(i*atoms_in_row, i*atoms_in_row + cut_col*atoms_in_lattice))
            cut_col_indices_list.append(np.arange((i+1)*atoms_in_row - cut_col*atoms_in_lattice, (i+1)*atoms_in_row))
    
        cut_col_indices = np.concatenate(cut_col_indices_list)
    
        cut_atom_positions.append(np.delete(posit_row_cut, cut_col_indices, axis = 1)) 
        cut_atom_params.append(np.delete(par_row_cut, cut_col_indices, axis = 1))
    
    return cut_atom_positions, cut_atom_params

def deep_unpack(atom_resolved_positions, atom_resolved_params, atom_num_list):

    deep_unpacked_positions = []
    deep_unpacked_params = []

    atom_type_num = len(atom_num_list)

    for i in range(atom_type_num):
        for j in range(atom_num_list[i]):

            indices = np.arange(j, (atom_resolved_positions[i].shape[-1]), atom_num_list[i])
            deep_unpacked_positions.append(atom_resolved_positions[i][:,indices])

            indices = np.arange(j, (atom_resolved_params[i].shape[-1]), atom_num_list[i])
            deep_unpacked_params.append(atom_resolved_params[i][:,indices])

    return deep_unpacked_positions,  deep_unpacked_params

def weight_average_poist_pix(deep_unpacked_positions, deep_unpacked_params, atom_num_list, weight = True):

    wa_posit_pix = []
    wa_params = []

    k = 0
    
    for i in range(len(atom_num_list)):

        posit = []
        params = []
        
        for j in range(atom_num_list[i]):

            if weight == True:

                posit.append(np.sum(deep_unpacked_positions[k]*deep_unpacked_params[k][0], axis = 1)/np.sum(deep_unpacked_params[k][0]))

            else: 

                posit.append(np.mean(deep_unpacked_positions[k], axis = 1))
            
            params.append(np.mean(deep_unpacked_params[k], axis = 1))

            k += 1

        wa_posit_pix.append(np.array(posit))
        wa_params.append(np.array(params))

    return wa_posit_pix, wa_params
