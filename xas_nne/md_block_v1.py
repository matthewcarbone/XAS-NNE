import os
from time import time
import numpy as np
import pathlib as path


path = 'D:\\BNL\\MD_Datasets\\benzene.xyz'
dir_path = 'D:\\BNL\\MD_Datasets\\'
file_ext = '.xyz'
file_name = 'benzene'


def split_file_on_blocks(file_path):

    # Context manager is better practice for this
    with open(file_path, "r") as input_file:

        lines = input_file.readlines()

        # Block separator is e.g. 12 for Benzene
        # Note you don't need the extra ``readline`` operation since you've
        # already loaded all of the file's lines into memory
        block_separator = lines[0]

        # --- Matt left off here ---
        block_id = ''
        blocks = []
        block_array =[]
        block_num = 0
        block_line =0

        for line in lines:
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
                #line_array.extend(line.rstrip().split())
                line_array.extend(line.rstrip().split()[0:4])
                block_array.append(line_array)

            block_line +=1

    return blocks


# This lets us input ``md_block_v1`` without actually running this code
if __name__ == '__main__':

    file_path = dir_path + file_name + file_ext
    blocks = split_file_on_blocks(file_path)

    # line_array.extend(line.rstrip().split()[0:4])

    print(len(blocks))
    print(len(blocks[10112]))
    print(blocks[10112])














