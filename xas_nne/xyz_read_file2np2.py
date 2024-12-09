import os
from time import time
import numpy as np
import pathlib as path


def process_xyz_list_to_flat(block_separator, xyz_list :list, columns_out:int):

    block_energy = ''
    blocks = []
    block_array =[]
    block_id = 0
    block_line =0

    for line in xyz_list:
        line_txt =line.rstrip()
        if(line_txt==block_separator):
            block_line = 1
            if(block_id !=0):
                blocks.append(block_array)
                block_array =[]
                block_id +=1
            else:
                block_id +=1
        elif(block_line ==2):
            block_energy = line.rstrip()
            print(block_energy)
        else:
            # extend
            line_array=[block_energy]
            line_array.extend(line.rstrip().split()[0:columns_out])
            block_array.append(line_array)

        block_line +=1

    return blocks

def process_xyz_list_to_structure(block_separator, xyz_list :list, columns_out:int):

    block_energy = ''
    blocks = []
    block_array =[]
    #The block ID is the number assigned to each snapshot and referenced in the first number in the arrays generated
    block_id_array = []
    #Indicates the atom
    block_header =[]
    #Indicates the coordinates
    lines_array =[]
    block_id = 0
    block_line =0

    for line in xyz_list:
        line_txt =line.rstrip()
        if(line_txt==block_separator):
            block_line = 1
            if(block_id !=0):
                block_id_array.append(block_id)
                block_array.append(block_id_array)
                block_array.append(block_header)
                block_array.append(lines_array)
                blocks.append(block_array)
                block_id_array = []
                block_header =[]
                lines_array =[]
                block_array =[]
                block_id +=1
            else:
                block_id +=1
        elif(block_line ==2):
            block_energy = line.rstrip()
        else:
            line_array=line.rstrip().split()[0:columns_out]
            lines_array.append(line_array[1:columns_out])
            block_header.append(line_array[0])
        block_line +=1

    return blocks


def process_xyz_file_to_array (input_dir, file_name, file_ext, columns_out:int):

    file_path = input_dir + file_name + file_ext
    with open(file_path, "r") as input_file:
        block_separator = input_file.readline().rstrip()
        lines= input_file.readlines()

    blocks = process_xyz_list_to_structure(block_separator, lines, columns_out)

    return blocks

def save_xyz_list_to_npz(output_dir, file_name, xyz_list):
    
    file_ext ='.npz'
    output_file_path =  output_dir + file_name + file_ext
    np.savez(output_file_path, xyz_list)


def process_xyz_dir(input_dir, output_dir, file_ext, file_names:list,  columns_out:int):

    for file_name in file_names:
        blocks = process_xyz_file_to_array(input_dir, file_name, file_ext, columns_out)
        save_xyz_list_to_npz(output_dir, file_name,blocks)

        print(len(blocks))
        print(len(blocks[10112]))
        print(blocks[10112])


# runner portion - specify parameters
input_dir = 'D:/BNL/MD_Datasets/'
output_dir = 'D:/BNL/MD_Datasets/data_out/'
file_ext = '.xyz'
file_names = ['ethanol']
columns_out =4

process_xyz_dir(input_dir, output_dir, file_ext, file_names,  columns_out)
