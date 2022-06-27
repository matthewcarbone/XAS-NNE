import os
import numpy as np

def md_block():

     path = 'D:\\BNL\\MD_Datasets\\benzene.xyz'

     input_file =open(path, 'r')
     block_separator =input_file.readline()

     lines= input_file.readlines()
     line_id =0
     block_id =1


     for line in lines:
          line_id += 1
          
     if(line_id==15):
          line_id=1
          block_id += 1
          print("Block {}".format(block_id))

     print("Block {}: Line {}: {}".format(block_id, line_id, line.strip()))

     coordinates = np.array([1,2], dtype=np.uint32)