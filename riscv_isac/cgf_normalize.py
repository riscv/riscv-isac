# See LICENSE.incore for details
from math import *
import riscv_isac.utils as utils
import itertools

def twos(val,bits):
    '''
    Finds the twos complement of the number
    :param val: input to be complemented
    :param bits: size of the input

    :type val: str or int
    :type bits: int

    :result: two's complement version of the input

    '''
    if isinstance(val,str):
        if '0x' in val:
            val = int(val,16)
        else:
            val = int(val,2)
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val

def sp_vals(bit_width,signed):
    if signed:
        conv_func = lambda x: twos(x,bit_width)
        sqrt_min = int(-sqrt(2**(bit_width-1)))
        sqrt_max = int(sqrt((2**(bit_width-1)-1)))
    else:
        sqrt_min = 0
        sqrt_max = int(sqrt((2**bit_width)-1))
        conv_func = lambda x: (int(x,16) if '0x' in x else int(x,2)) if isinstance(x,str) else x

    dataset = [3, "0x"+"".join(["5"]*int(bit_width/4)), "0x"+"".join(["a"]*int(bit_width/4)), 5, "0x"+"".join(["3"]*int(bit_width/4)), "0x"+"".join(["6"]*int(bit_width/4))]
    dataset = list(map(conv_func,dataset)) + [int(sqrt(abs(conv_func("0x8"+"".join(["0"]*int((bit_width/4)-1)))))*(-1 if signed else 1))] + [sqrt_min,sqrt_max]
    return dataset + [x - 1 if x>0 else 0 for x in dataset] + [x+1 for x in dataset]

def sp_dataset(bit_width,var_lst=["rs1_val","rs2_val"],signed=True):
    coverpoints = []
    datasets = []
    var_names = []
    for var in var_lst:
        if isinstance(var,tuple) or isinstance(var,list):
            if len(var) == 3:
                var_sgn = var[2]
            else:
                var_sgn = signed
            var_names.append(var[0])
            datasets.append(sp_vals(int(var[1]),var_sgn))
        else:
            var_names.append(var)
            datasets.append(sp_vals(bit_width,signed))
    dataset = itertools.product(*datasets)
    for entry in dataset:
        coverpoints.append(' and '.join([var_names[i]+"=="+str(entry[i]) for i in range(len(var_names))]))
    return coverpoints

def walking_ones(var, size, signed=True, fltr_func=None, scale_func=None):
    '''
    This function converts an abstract walking-ones function into individual
    coverpoints that can be used by ISAC. The unrolling of the function accounts
    of the size, sign-bit, filters and scales. The unrolled coverpoints will
    contain a pattern a single one trickling down from LSB to MSB. The final
    coverpoints can vary depending on the filtering and scaling functions

    :param var: input variable that needs to be assigned the coverpoints
    :param size: size of the bit-vector to generate walking-1s
    :param signed: when true indicates that the unrolled points be treated as signed integers.
    :param fltr_func: a lambda function which defines a filtering routine to keep only a certain values from the unrolled coverpoints
    :param scale_func: a lambda function which defines the scaling that should be applied to the unrolled coverpoints that have been generated.

    :type var: str
    :type size: int
    :type signed: bool
    :type fltr_func: function
    :type scale_func: function
    :result: dictionary of unrolled filtered and scaled coverpoints
    '''
    if not signed:
        dataset = [1 << exp for exp in range(size)]
    else:
        dataset = [twos(1 << exp,size) for exp in range(size)]
    if scale_func:
        dataset = [scale_func(x) for x in dataset]
    if fltr_func:
        dataset = filter(fltr_func,dataset)
    coverpoints = [var + ' == ' + str(d) for d in dataset]
    return coverpoints

def walking_zeros(var, size,signed=True, fltr_func=None, scale_func=None):
    '''
    This function converts an abstract walking-zeros function into individual
    coverpoints that can be used by ISAC. The unrolling of the function accounts
    of the size, sign-bit, filters and scales. The unrolled coverpoints will
    contain a pattern a single zero trickling down from LSB to MSB. The final
    coverpoints can vary depending on the filtering and scaling functions

    :param var: input variable that needs to be assigned the coverpoints
    :param size: size of the bit-vector to generate walking-1s
    :param signed: when true indicates that the unrolled points be treated as signed integers.
    :param fltr_func: a lambda function which defines a filtering routine to keep only a certain values from the unrolled coverpoints
    :param scale_func: a lambda function which defines the scaling that should be applied to the unrolled coverpoints that have been generated.

    :type var: str
    :type size: int
    :type signed: bool
    :type fltr_func: function
    :type scale_func: function
    :result: dictionary of unrolled filtered and scaled coverpoints
    '''
    mask = 2**size -1
    if not signed:
        dataset = [(1 << exp)^mask for exp in range(size)]
    else:
        dataset = [twos((1 << exp)^mask,size) for exp in range(size)]
    if scale_func:
        dataset = [scale_func(x) for x in dataset]
    if fltr_func:
        dataset = filter(fltr_func,dataset)
    coverpoints = [var + ' == ' + str(d) for d in dataset]
    return coverpoints

def alternate(var, size, signed=True, fltr_func=None,scale_func=None):
    '''
    This function converts an abstract alternate function into individual
    coverpoints that can be used by ISAC. The unrolling of the function accounts
    of the size, sign-bit, filters and scales. The unrolled coverpoints will
    contain a pattern of alternating 1s and 0s. The final
    coverpoints can vary depending on the filtering and scaling functions

    :param var: input variable that needs to be assigned the coverpoints
    :param size: size of the bit-vector to generate walking-1s
    :param signed: when true indicates that the unrolled points be treated as signed integers.
    :param fltr_func: a lambda function which defines a filtering routine to keep only a certain values from the unrolled coverpoints
    :param scale_func: a lambda function which defines the scaling that should be applied to the unrolled coverpoints that have been generated.

    :type var: str
    :type size: int
    :type signed: bool
    :type fltr_func: function
    :type scale_func: function

    :result: dictionary of unrolled filtered and scaled coverpoints
    '''
    t1 =( '' if size%2 == 0 else '1') + ''.join(['01']*int(size/2))
    t2 =( '' if size%2 == 0 else '0') + ''.join(['10']*int(size/2))
    if not signed:
        dataset = [int(t1,2),int(t2,2)]
    else:
        dataset = [twos(t1,size),twos(t2,size)]
    if scale_func:
        dataset = [scale_func(x) for x in dataset]
    if fltr_func:
        dataset = filter(fltr_func,dataset)
    coverpoints = [var + ' == ' + str(d) for d in dataset]
    return coverpoints

def simd_val_comb(xlen, bit_width, var_lst=["rs1_val","rs2_val"], signed=True):
    '''
    This function coverts an rs1_val and rs2_val combination coverpoints.
    '''
    twocompl_offset = 1 << bit_width
    mask = twocompl_offset - 1
    simd_elm_positions = [i * bit_width for i in range(xlen//bit_width)]
    coverpoints = []
    for s in simd_elm_positions:
        rs1_element_bits = f'(({var_lst[0]} >> {s}) & {mask})'
        rs2_element_bits = f'(({var_lst[1]} >> {s}) & {mask})'
        #convert unsigned {bit_width} bits to signed values in native width
        rs1_positive_elem_value = f'{rs1_element_bits}'
        rs1_negative_elem_value = f'({rs1_element_bits} - {twocompl_offset})'
        #convert unsigned {bit_width} bits to signed values in native width
        rs2_positive_elem_value = f'{rs2_element_bits}'
        rs2_negative_elem_value = f'({rs2_element_bits} - {twocompl_offset})'
        if signed:
            coverpoints += [f"{rs1_positive_elem_value} > 0 and {rs2_positive_elem_value} > 0"]
            coverpoints += [f"{rs1_positive_elem_value} > 0 and {rs2_negative_elem_value} < 0"]
            coverpoints += [f"{rs1_negative_elem_value} < 0 and {rs2_positive_elem_value} > 0"]
            coverpoints += [f"{rs1_negative_elem_value} < 0 and {rs2_negative_elem_value} < 0"]
            coverpoints += [f"{rs1_positive_elem_value} == {rs2_positive_elem_value}"]
            coverpoints += [f"{rs1_negative_elem_value} == {rs2_negative_elem_value}"]
            coverpoints += [f"{rs1_positive_elem_value} != {rs2_positive_elem_value}"]
            coverpoints += [f"{rs1_negative_elem_value} != {rs2_negative_elem_value}"]
        else:
            coverpoints += [f"{rs1_element_bits} == {rs2_element_bits}"]
            coverpoints += [f"{rs1_element_bits} != {rs2_element_bits}"]
    return coverpoints

def simd_base_val(var, xlen, bit_width, signed=True):
    '''
    This function coverts an rs_val base data coverpoints, here refer to the original 32/64 bit constraint
    and modify it to the simd 8/16 bit and signed/unsigned bit constraint.
    '''
    sgn_dataset_pos = ['0', '1', str(2**(bit_width-1)-1)]
    sgn_dataset_neg = ['-1', str(-2**(bit_width-1))]
    usgn_dataset = ['0', '1', str(2**(bit_width)-1)]
    twocompl_offset = 1 << bit_width
    mask = twocompl_offset - 1
    simd_elm_positions = [i * bit_width for i in range(xlen//bit_width)]
    coverpoints = []

    for s in simd_elm_positions:
        element_bits = f'(({var} >> {s}) & {mask})'
        #convert unsigned {bit_width} bits to signed values in native width
        positive_elem_value = f'{element_bits}'
        negative_elem_value = f'({element_bits} - {twocompl_offset})'

        if signed:
            coverpoints += [f"{positive_elem_value} == {sgn_dataset_pos[i]}" for i in range(len(sgn_dataset_pos))]
            coverpoints += [f"{negative_elem_value} == {sgn_dataset_neg[i]}" for i in range(len(sgn_dataset_neg))]
        else:
            coverpoints += [f"{positive_elem_value} == {usgn_dataset[i]}" for i in range(len(usgn_dataset))]
    return coverpoints

def simd_sp_dataset(xlen, bit_width,var_lst=["rs1_val","rs2_val"],signed=True):
    '''
    This function coverts an rs1_val and rs2_val special data coverpoints,
    here refer to the original 32/64 bit constraint and modify it to the
    simd 8/16 bit and signed/unsigned bit constraint.
    '''
    coverpoints = []
    datasets = []
    var_names = []
    twocompl_offset = 1 << bit_width
    mask = twocompl_offset - 1
    simd_elm_positions = [i * bit_width for i in range(xlen//bit_width)]

    for var in var_lst:
        if isinstance(var,tuple) or isinstance(var,list):
            if len(var) == 3:
                var_sgn = var[2]
            else:
                var_sgn = signed
            var_names.append(var[0])
            datasets.append(sp_vals(int(var[1]),var_sgn))
        else:
            var_names.append(var)
            datasets.append(sp_vals(bit_width,signed))
    dataset = itertools.product(*datasets)

    for data in dataset:
        for s in simd_elm_positions:
            rs1_element_bits = f'(({var_names[0]} >> {s}) & {mask})'
            rs2_element_bits = f'(({var_names[1]} >> {s}) & {mask})'
            #convert unsigned {bit_width} bits to signed values in native width
            rs1_positive_elem_value = f'{rs1_element_bits}'
            rs1_negative_elem_value = f'({rs1_element_bits} - {twocompl_offset})'
            #convert unsigned {bit_width} bits to signed values in native width
            rs2_positive_elem_value = f'{rs2_element_bits}'
            rs2_negative_elem_value = f'({rs2_element_bits} - {twocompl_offset})'
            if signed:
                rs1_elem_value = rs1_positive_elem_value if (data[0] >= 0) else rs1_negative_elem_value
                rs2_elem_value = rs2_positive_elem_value if (data[1] >= 0) else rs2_negative_elem_value
                coverpoints += [f"{rs1_elem_value} == {data[0]} and {rs2_elem_value} == {data[1]}"]
            else:
                coverpoints += [f"{rs1_positive_elem_value} == {data[0]} and {rs2_positive_elem_value} == {data[1]}"]
    return coverpoints

def simd_clip(var,var_imm,var_imm_width, xlen, bit_width, signed=True):
    '''
    This function converts an rs1_val and imm_val into individual coverpoints that can be used by ISAC.
    '''
    twocompl_offset = 1 << bit_width
    mask = twocompl_offset - 1
    sign_bit_mask = 1 << (bit_width - 1)
    simd_elm_positions = [i * bit_width for i in range(xlen//bit_width)]
    coverpoints = []
    for s in (simd_elm_positions):
        for val_imm in range (2**var_imm_width):
            positive_clip_threshold = 2**val_imm - 1
            negative_clip_threshold = -2**val_imm
            #Divided into 4 blocks
            element_bits = f'(({var} >> {s}) & {mask})'
            is_positive_value = f'{element_bits} < {sign_bit_mask}'
            #convert unsigned {bit_width} bits to signed values in native width
            positive_elem_value = f'{element_bits}'
            negative_elem_value = f'{element_bits}-{twocompl_offset}'
            if signed and val_imm != bit_width - 1 or not signed:
                coverpoints += [f"{positive_elem_value} >  {positive_clip_threshold} and {var_imm} == {val_imm} and {is_positive_value}"]
                coverpoints += [f"{negative_elem_value} <  {negative_clip_threshold} and {var_imm} == {val_imm} and not {is_positive_value}"]
            if val_imm != 0 and signed:
                coverpoints += [f"{positive_elem_value} <  {positive_clip_threshold} and {var_imm} == {val_imm} and {is_positive_value}"]
                coverpoints += [f"{negative_elem_value} >  {negative_clip_threshold} and {var_imm} == {val_imm} and not {is_positive_value}"]
            if signed:
                coverpoints += [f"{positive_elem_value} == {positive_clip_threshold} and {var_imm} == {val_imm} and {is_positive_value}"]
                coverpoints += [f"{negative_elem_value} == {negative_clip_threshold} and {var_imm} == {val_imm} and not {is_positive_value}"]
    return coverpoints

def expand_cgf(cgf_files, xlen):
    '''
    This function will replace all the abstract functions with their unrolled
    coverpoints

    :param cgf_files: list of yaml file paths which together define the coverpoints
    :param xlen: XLEN of the riscv-trace

    :type cgf: list
    :type xlen: int
    '''
    cgf = utils.load_cgf(cgf_files)
    for labels, cats in cgf.items():
        if labels != 'datasets':
            for label,node in cats.items():
                if isinstance(node,dict):
                    if 'abstract_comb' in node:
                        temp = cgf[labels][label]['abstract_comb']
                        del cgf[labels][label]['abstract_comb']
                        for coverpoints, coverage in temp.items():
                                if 'walking' in coverpoints or 'alternate' in coverpoints or 'sp_dataset' in coverpoints:
                                    exp_cp = eval(coverpoints)
                                    for e in exp_cp:
                                        cgf[labels][label][e] = coverage
    return cgf


