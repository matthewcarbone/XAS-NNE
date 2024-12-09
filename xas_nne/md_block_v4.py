import os
from time import time
import numpy as np
import pathlib as path
​
​
def process_xyz_list(block_separator, xyz_list :list, columns_out:int):
​
    block_id = ''
    blocks = []
    block_array =[]
    block_num = 0
    block_line =0
​
    for line in xyz_list:
        line_txt =line.rstrip()
        if(line_txt==block_separator):
            block_line = 1
            if(block_num !=0):
                blocks.append(block_array)
                block_array =[]
                block_num +=1
            else:
                block_num +=1
        elif(block_line ==2):
            block_id = line.rstrip()
            print(block_id)
        else:
            line_array=[block_id]
            line_array.extend(line.rstrip().split()[0:columns_out])
            block_array.append(line_array)
​
        block_line +=1
​
    return blocks
​
​
def process_xyz_file_to_array (input_dir, file_name, file_ext, columns_out:int):
​
    file_path = input_dir + file_name + file_ext
    with open(file_path, "r") as input_file:
        block_separator = input_file.readline().rstrip()
        lines= input_file.readlines()
​
    blocks = process_xyz_list(block_separator, lines, columns_out)
​
    return blocks
​
def save_xyz_list_to_npz(output_dir, file_name, xyz_list):
​
    file_ext ='.npz'
    output_file_path =  output_dir + file_name + file_ext
    np.savez(output_file_path, xyz_list)
​
    return
​
def process_xyz_dir(input_dir, output_dir, file_ext, file_names:list,  columns_out:int):
​
    for file_name in file_names:
        blocks = process_xyz_file_to_array(input_dir, file_name, file_ext, columns_out)
        save_xyz_list_to_npz(output_dir, file_name,blocks)
​
        print(len(blocks))
        print(len(blocks[10112]))
        print(blocks[10112])
​
    return
​
# runner portion - specify parameters
input_dir = 'D:/mike/bnl/data/'
output_dir = 'D:/mike/bnl/data_out/'
file_ext = '.xyz'
file_names = ['benzene','ethanol']
columns_out =4
​
process_xyz_dir(input_dir, output_dir, file_ext, file_names,  columns_out)