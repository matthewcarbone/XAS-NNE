import os

def md_xyz_block():

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

    atoms = []
    coordinates = []
    xyz = open(file)
    n_atoms = int(xyz.readline())
    title = xyz.readline()
    for line in xyz:
        line = file.readline().replace(".*^", "e").replace("*^", "e")
        atom,x,y,z = line.split()
        atoms.append(str(line[0]))
        coordinates.append(np.array(line[1:4], dtype=float))

    coordinates = np.array(coordinates)
    xyz.close()
    np.save(coordinates)