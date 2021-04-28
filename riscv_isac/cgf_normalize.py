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

def simd_val_comb(xlen, bit_width, signed=True):
    fmt = {8: 'b', 16: 'h', 32: 'w', 64: 'd'}
    sz = fmt[bit_width]
    var_num = xlen//bit_width
    coverpoints = []
    for i in range(var_num):
        var1 = f'rs1_{sz}{i}_val'
        var2 = f'rs2_{sz}{i}_val'
        if (signed):
            coverpoints += [f'{var1} > 0 and {var2} > 0']
            coverpoints += [f'{var1} > 0 and {var2} < 0']
            coverpoints += [f'{var1} < 0 and {var2} < 0']
            coverpoints += [f'{var1} < 0 and {var2} > 0']
            coverpoints += [f'{var1} == {var2}']
            coverpoints += [f'{var1} != {var2}']
        else:
            coverpoints += [f'{var1} == {var2} and {var1} > 0 and {var2} > 0']
            coverpoints += [f'{var1} != {var2} and {var1} > 0 and {var2} > 0']
    return coverpoints

def simd_base_val(rs, xlen, bit_width, signed=True):
    fmt = {8: 'b', 16: 'h', 32: 'w', 64: 'd'}
    sz = fmt[bit_width]
    var_num = xlen//bit_width
    sign_val = [(-2**(bit_width-1)), -1, 0, 1, int((2**(bit_width-1)-1))]
    usign_val = [0, 1, 2**bit_width-2, 2**bit_width-1]
    coverpoints = []
    for i in range(var_num):
        var = f'{rs}_{sz}{i}_val'
        if (signed):
            for val in sign_val:
                coverpoints += [f'{var} == {val}']
        else:
            for val in usign_val:
                coverpoints += [f'{var} == {val}']
    return coverpoints

def simd_imm_val(imm, bit_width):
    usign_val = (2**(bit_width))
    coverpoints = []
    for i in range(usign_val):
        coverpoints += [f'{imm} == {i}']
    return coverpoints

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
                                if 'walking' in coverpoints or 'alternate' in coverpoints or 'sp_dataset' in coverpoints or 'simd' in coverpoints:
                                    exp_cp = eval(coverpoints)
                                    for e in exp_cp:
                                        cgf[labels][label][e] = coverage
    return cgf
