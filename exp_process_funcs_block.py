from email.mime import image
import os
import numpy as np
import  matplotlib.pyplot as plt
import cv2
from PIL import Image
import PIL
import torch
import einops
import re
import random

def center_crop_size(img,final_size=[256,256]):
    raw_shape=np.array(img.shape)
    mv =  np.subtract(raw_shape,np.array(final_size))/2
    l_s1=int(mv[0])
    l_e1=l_s1+int(final_size[0])
    l_s2=int(mv[1])
    l_e2=l_s2+int(final_size[1])  
    img_crop = img[l_s1:l_e1,l_s2:l_e2]
    return img_crop


#define a function use above section and assume all the size in AFM image is same,
# to be notice the x and y here means height and width, which is different from the normal definition, and hence 
# flag [x,y] means height diriction and witdh dirction x:1 top->down y:1 left->right
def slide_cut(IMG,step,save_path="./",flag=[1,1],L_exp=np.array([2.5,2.5]),L_tar=2.5,stride=0):
    flag_x , flag_y = flag
    img_test = IMG[0]
    img_shape = np.array(img_test.shape)
    box_pix = np.divide(img_shape*L_tar,L_exp)
    box_pix=box_pix.astype(int)
    stride_pix = np.divide(img_shape*stride,L_exp)
    stride_pix=stride_pix.astype(int)
    for dx in range(step[0]):
        for dy in range(step[1]):
            bx = [flag_x*dx*stride_pix[0],flag_x*(dx*stride_pix[0]+box_pix[0])]
            by = [flag_y*dy*stride_pix[1],flag_y*(dy*stride_pix[1]+box_pix[1])]
            x_l,x_h = sorted(bx)
            y_l,y_h = sorted(by)
            dic_name = "{}_{}_{}_{}".format(flag_x,flag_y,dx,dy)
            dic_path=os.path.join(save_path,dic_name)
            if not os.path.exists(dic_path):
                os.mkdir(dic_path)
            for ind, img_test in enumerate(IMG):
                if x_h == 0 and y_h==0:
                    img_new = img_test[x_l:,y_l:]
                elif x_h==0:
                    img_new = img_test[x_l:,y_l:y_h]
                elif y_h==0:
                    img_new = img_test[x_l:x_h,y_l:]
                else:
                    img_new = img_test[x_l:x_h,y_l:y_h]
                img1 = img_new.copy()
                fn_write=os.path.join(dic_path, "{}.png".format(ind))
                cv2.imwrite(fn_write,img1)


def i_delta(x):
    if x<0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def cutoff(r,rc=16):
    # r and rc in cell legnth
    if r< rc:
        return 0.5*(np.cos(np.pi*r/rc)+1)
    else:
        return 0 

def generate_ratio_map(shape=[32,32]):
    # generate a confident ratio map, the maximum is the center
    ratio_map = np.ones(shape,dtype=float)
    center = np.array(shape)/2
    rc = int(np.power((center[0]**2+center[1]**2),0.5))+1
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.power(((i-center[0])**2+(j-center[1])**2),0.5)
            ratio_map[i,j]=cutoff(r,rc=rc)
    return ratio_map
    

# combine the npy data into a data set
def combine_cell(file_list,flag_list,L_exp,L_tar,stride,l_cell):
    '''
        file_list : the files need to combine
        flag_list: the flag of the sub image when and where to cut : flag_h, flag_w, dh, dw
        seg_size: the segmentation image size in pixels
        L_exp: the size of the raw image in nm,
        L_tar: the size of the sub image in nm,
        stride: the step legnth for cut in nm.
        l_cell: the cell legnth in nm, here is 2.5/32
    '''
    L_cubic =L_exp/l_cell # 0: height, ind 1: width
    L_cubic=L_cubic.astype(int)
    L_cubic=L_cubic+1
    Cubic_combine = np.zeros([L_cubic[1],L_cubic[0],4,8],dtype=float) # Oxygen:dx,dy,dz,confidence, Hydrogen...
    Book = np.zeros_like(Cubic_combine)
    r_m = generate_ratio_map()
    for ind, fn in enumerate(file_list):
        target_temp: torch.Tensor = np.load(fn)
        target_temp = einops.rearrange(target_temp[...,(2,3,1,0)],'Z X Y E C-> X Y Z (E C)')
        flag_y,flag_x,dy,dx = flag_list[ind]
        flag_y=-1*flag_y
        local_pos = [flag_x*stride*dx+i_delta(flag_x)*(L_exp[1]-L_tar),
                     flag_y*stride*dy+i_delta(flag_y)*(L_exp[0]-L_tar),0]
        local_pos_c = np.array(local_pos)/l_cell
        for ci in range(32):
            for cj in range(32):
                for ck in range(4):
                    for ca in range(2):
                        c_index = np.array([ci,cj,ck])
                        cline = target_temp[ci,cj,ck,4*ca:4*(ca+1)]
                        c_offset = cline[:3]
                        c_confidence = cline[3]
                        #if we count the negative confidence, we may need do sigmoid after the calculation
                        if c_confidence < 0:
                            continue
                        else:
                            ratio = r_m[ci,cj]
                            c_confidence = sigmoid(c_confidence)
                            c_final_loc = np.add(np.add(local_pos_c,c_index),c_offset)
                            add_index = c_final_loc.astype(int)
                            ax,ay,az = add_index
                            if c_confidence >=0.5:
                                add_offset = np.subtract(c_final_loc,add_index)
                            else:
                                add_offset=np.zeros_like(add_index,dtype=float)
                            #if ax==45 and ay==40 and az==2:
                            #    print(add_offset, c_confidence, ratio)
                            add_info = np.append(add_offset,(c_confidence-0.5))*ratio
                            add_book = np.ones_like(add_info)*ratio
                            if az >= 4:
                                continue
                            Cubic_combine[ax,ay,az,4*ca:4*(ca+1)] = np.add(Cubic_combine[ax,ay,az,4*ca:4*(ca+1)],add_info)
                            Book[ax,ay,az,4*ca:4*(ca+1)]=np.add(Book[ax,ay,az,4*ca:4*(ca+1)],add_book)
    #Cubic_combine[...,3] = sigmoid(Cubic_combine[...,3])
    #Cubic_combine[...,7] = sigmoid(Cubic_combine[...,7])
    #Cubic_combine[...,3] = Cubic_combine[...,3]+0.5
    #Cubic_combine[...,7] = Cubic_combine[...,7]+0.5
    Book=np.where(Book==0.0,1.0,Book)
    Cubic_out = np.divide(Cubic_combine,Book)
    return Cubic_out

def nms(position, ele, lattice):
    #ele2r = {'H': 0.528, 'O': 0.74}
    ele2r = {'H': 1.0, 'O': 2}
    r = ele2r[ele]
    if len(position) == 0:
        return position
    position = np.asarray(sorted(position, key=lambda x: x[-1], reverse=True))
    mask = np.full(len(position), True)
    mask[1000:] = False  # 最多1000个原子，截断
    random.shuffle(mask)
    for i in range(len(position)):
        if mask[i]:
            for j in range(i + 1, len(position)):
                if mask[j]:
                    distance = np.sqrt(np.sum(np.square((position[i][:3] - position[j][:3]).dot(lattice))))
                    if distance < r:
                        mask[j] = False
    position = position[mask]
    return position

def target2positions(target, info, ele_name, threshold=0, NMS=False):
    positions = {}
    for i, ele in enumerate(ele_name):
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > threshold]
        if NMS:
            position = nms(position, ele, info['lattice'])
        positions[ele] = position
    return positions

def positions2poscar(positions, info, path_prediction):
    with open(path_prediction, 'w') as file:
        file.write(str(info['comment']))
        file.write(str(info['scale']) + '\n')
        lattice = info["lattice"]
        for i in range(3):
            file.write(f'  \t{lattice[i, 0]:.8f} {lattice[i, 1]:.8f} {lattice[i, 2]:.8f}\n')
        line1 = '\t'
        line2 = '\t'
        for ele in positions.keys():
            position = positions[ele]
            line1 += str(ele) + ' '
            line2 += str(len(position)) + ' '
            try:
                position_array = np.concatenate((position_array, position), axis=0)
            except UnboundLocalError:
                position_array = position
        line1 += '\n'
        line2 += '\n'
        file.write(line1)
        file.write(line2)
        file.write("Selective dynamics\nDirect\n")
        for line in position_array:
            file.write(f' {line[0]:.8f} {line[1]:.8f} {line[2]:.8f} T T T\n')

# directly read the poscar and then combine to avoid the nms
def generate_target_abs(info, positions, N):
    ele_name = ['O', 'H']
    size = (32, 32, N)
    targets = np.zeros(size + (4 * len(ele_name),))
    for i, ele in enumerate(ele_name):
        target = np.zeros(size + (4,))
        position = positions[ele]
        position = position.dot(np.diag(size))
        for j in range(info['ele_num'][ele]):
            pos = position[j]
            coordinate = np.int_(pos)
            offset = pos - coordinate
            idx_x, idx_y, idx_z = coordinate
            offset_x, offset_y, offset_z = offset
            if idx_z >= N:
                idx_z = N - 1
                offset_z = 1 - 1e-4
            if target[idx_x, idx_y, idx_z, 3] == 0.0:
                target[idx_x, idx_y, idx_z] = [offset_x, offset_y, offset_z, 1.0]
            else:
                pass
                #raise Exception
        targets[..., 4 * i: 4 * (i + 1)] = target
    return targets

def clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp

def read_POSCAR(file_name):
    """
    read the POSCAR or CONTCAR of VASP FILE
    and return the data position
    """
    with open(file_name) as fr:
        comment = fr.readline()
        line = fr.readline()
        scale_length = float(clean(line)[0])
        lattice = []
        for i in range(3):
            lattice.append(clean(fr.readline()).astype(float))
        lattice = np.array(lattice)
        ele_name = clean(fr.readline())
        counts = clean(fr.readline()).astype(int)
        ele_num = dict(zip(ele_name, counts))
        fr.readline()
        fr.readline()
        positions = {}
        for ele in ele_name:
            position = []
            for _ in range(ele_num[ele]):
                line = clean(fr.readline())
                position.append(line[:3].astype(float))
            positions[ele] = np.asarray(position)
    info = {'comment': comment, 'scale': scale_length, 'lattice': lattice, 'ele_num': ele_num,
            'ele_name': tuple(ele_name)}
    return info, positions

# combine the poscar data into a data set
def combine_cell_pos(file_list,flag_list,L_exp,L_tar,stride,l_cell):
    '''
        file_list : the files need to combine
        flag_list: the flag of the sub image when and where to cut : flag_h, flag_w, dh, dw
        seg_size: the segmentation image size in pixels
        L_exp: the size of the raw image in nm,
        L_tar: the size of the sub image in nm,
        stride: the step legnth for cut in nm.
        l_cell: the cell legnth in nm, here is 2.5/32
    '''
    L_cubic =L_exp/l_cell # 0: height, ind 1: width
    L_cubic=L_cubic.astype(int)
    L_cubic=L_cubic+1
    print(L_cubic)
    Cubic_combine = np.zeros([L_cubic[1],L_cubic[0],4,8],dtype=float) # Oxygen:dx,dy,dz,confidence, Hydrogen...
    Book = np.zeros_like(Cubic_combine)
    r_m = generate_ratio_map()
    for ind, fn in enumerate(file_list):
        #print(fn)
        t_info, t_positions = read_POSCAR(fn)
        target_temp = generate_target_abs(t_info,t_positions,4)
        flag_y,flag_x,dy,dx = flag_list[ind]
        flag_y=-1*flag_y
        local_pos = [flag_x*stride*dx+i_delta(flag_x)*(L_exp[1]-L_tar),
                     flag_y*stride*dy+i_delta(flag_y)*(L_exp[0]-L_tar),0]
        local_pos_c = np.array(local_pos)/l_cell
        for ci in range(32):
            for cj in range(32):
                for ck in range(4):
                    for ca in range(2):
                        c_index = np.array([ci,cj,ck])
                        cline = target_temp[ci,cj,ck,4*ca:4*(ca+1)]
                        c_offset = cline[:3]
                        c_confidence = cline[3]
                        #there is no negative confidence, no need for sigmoid, read poscar directly
                        if c_confidence < 0.5:
                            continue
                        else:
                            ratio = r_m[ci,cj]
                            #c_confidence = sigmoid(c_confidence)
                            c_final_loc = np.add(np.add(local_pos_c,c_index),c_offset)
                            #print(c_final_loc)
                            add_index = c_final_loc.astype(int)
                            ax,ay,az = add_index
                            if c_confidence >=0.5:
                                add_offset = np.subtract(c_final_loc,add_index)
                            else:
                                add_offset=np.zeros_like(add_index,dtype=float)
                            #if ax==45 and ay==40 and az==2:
                            #    print(add_offset, c_confidence, ratio)
                            add_info = np.append(add_offset,(c_confidence-0.5))*ratio
                            add_book = np.ones_like(add_info)*ratio
                            Cubic_combine[ax,ay,az,4*ca:4*(ca+1)] = np.add(Cubic_combine[ax,ay,az,4*ca:4*(ca+1)],add_info)
                            Book[ax,ay,az,4*ca:4*(ca+1)]=np.add(Book[ax,ay,az,4*ca:4*(ca+1)],add_book)
    #Cubic_combine[...,3] = sigmoid(Cubic_combine[...,3])
    #Cubic_combine[...,7] = sigmoid(Cubic_combine[...,7])
    #Cubic_combine[...,3] = Cubic_combine[...,3]+0.5
    #Cubic_combine[...,7] = Cubic_combine[...,7]+0.5
    Book=np.where(Book==0.0,1.0,Book)
    Cubic_out = np.divide(Cubic_combine,Book)
    return Cubic_out

def nms_O(position, ele, info):
    #ele2r = {'H': 0.528, 'O': 0.74}
    ele2r = {'H': 0.528, 'O': 2}
    r = ele2r[ele]
    if len(position) == 0:
        return position
    position = np.asarray(sorted(position, key=lambda x: x[-1], reverse=True))
    mask = np.full(len(position), True)
    mask[1000:] = False  # 最多1000个原子，截断
    random.shuffle(mask)
    for i in range(len(position)):
        if mask[i]:
            for j in range(i + 1, len(position)):
                if mask[j]:
                    distance = np.sqrt(np.sum(np.square((position[i][:3] - position[j][:3]).dot(info['lattice']))))
                    if distance < r:
                        mask[j] = False
    position = position[mask]
    return position

# TODO:有问题
def target2positions_O(target, info, ele_name, threshold=0, NMS=False):
    bond_range = (0.8, 1.2)
    positions = {}
    for i, ele in enumerate(ele_name):
        if ele != 'O':
            continue
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > threshold]
        if NMS:
            position = nms_O(position, ele, info)
        positions[ele] = position
    for i, ele in enumerate(ele_name):
        if ele != 'H':
            continue
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > -3]
        if NMS:
            position = nms_O(position, ele, info)
        H_list = []
        for pos_O in positions['O']:
            distance = np.sqrt(np.sum(np.square((pos_O[None, :3] - position[:, :3]).dot(info['lattice'])), axis=1))
            mask = (distance > 0.8) & (distance < 1.2)
            position_H = position[mask]
            position_H = sorted(position_H, key=lambda x: x[-1], reverse=True)
            for pos_H in position_H[:2]:
                H_list.append(pos_H)
        positions[ele] = np.array(H_list)
    return positions

def trans_list(line):
    data0=[]
    line0=line.strip().replace('\t',' ')
    line1=re.findall(r"[\[](.*?)[\]]",line0)
    list2=line1[0].split(' ')
    for i in list2:
        if i!= '':
            data0.append(i)
    temp=np.array(data0)
    return temp


def read_slide_info(fp):
    with open(fp) as fr:
        fr.readline()
        line_exp = fr.readline().strip()
        r_exp = np.array(line_exp.split())
        fr.readline()
        line_step = fr.readline().strip()
        r_step= np.array(line_step.split())
        fr.readline()
        r_tar=float(fr.readline().strip())
        fr.readline()
        r_stride=float(fr.readline().strip())
        
    return r_exp.astype(float),r_step.astype(int), r_tar,r_stride