import numpy as np
import matplotlib.pyplot as plt

from ovrolwasolar import deconvolve as odeconv
from ovrolwasolar import flagging as oflag
import copy
from casatools import msmetadata, ms, table, quanta, measures
from astropy.io import fits


# prepare array and files

fname_fast_src = '/data07/peijinz/test-imaging/testdata/fast/20240519_173004_55MHz.ms'
fname_slow_src = '/data07/peijinz/test-imaging/testdata/slow/20240519_173002_55MHz_calibrated_selfcalibrated_sun_only.ms'
badant_csv = '/data07/peijinz/test-imaging/testdata/slow/20240519_173002_55MHz.badants'

# get the name of antenna from the header
msmd = msmetadata()
msmd.open(fname_fast_src)
fast_antenna_names = msmd.antennanames()
msmd.close()
msmd.open(fname_slow_src)
slow_antenna_names = msmd.antennanames()
msmd.close()

def compose_str_to_output(snr, old_set, new_set, replaced, new_in):
    return (str(snr)+','+ ';'.join([str(idx) for idx in old_set]) + 
                    ',' + ';'.join([str(idx) for idx in new_set]) + 
                    ',' + ';'.join([str(idx) for idx in replaced]) + 
                    ',' + ';'.join([str(idx) for idx in new_in]))

def convert_str_to_var(s):
    snr, old_set, new_set, replaced, new_in = s.split(',')
    return (float(snr), [int(i) for i in old_set.split(';')], [int(i) for i in new_set.split(';')], 
                        [int(i) for i in replaced.split(';')], [int(i) for i in new_in.split(';')])

# read badant csv file
badant = np.loadtxt(badant_csv, delimiter=',', dtype='str')
badant = np.int32(badant)
bad_ant_names = [slow_antenna_names[int(i)] for i in badant]

# create a tmp dir and copy the ms files to the tmp
import os
import shutil
tmp_dir = '/data07/peijinz/test-imaging/testdata/tmp'
# rm the tmp dir if it exists
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
# copy the ms folder to the tmp dir
shutil.copytree(fname_fast_src, os.path.join(tmp_dir, os.path.basename(fname_fast_src)))
shutil.copytree(fname_slow_src, os.path.join(tmp_dir, os.path.basename(fname_slow_src)))

fname_fast = os.path.join(tmp_dir, os.path.basename(fname_fast_src))
fname_slow = os.path.join(tmp_dir, os.path.basename(fname_slow_src))


# save the flagging for future use 

tb = table()
tb.open(fname_slow)
flagging_src = tb.getcol('FLAG')
tb.close()
# save the flagging to np file
np.save('/data07/peijinz/test-imaging/testdata/flagging.npy', flagging_src)



#init the ant_pool
ant_pool_to_iterate = np.setdiff1d(slow_antenna_names, bad_ant_names)
current_set_48 = copy.deepcopy(fast_antenna_names)


n_iter  = 100


for idx in range(n_iter):

    old_set48_idx = np.array([slow_antenna_names.index(i) for i in current_set_48])

    # random select 10 antennas from the slow which is not in the fast, replace random 10 antennas in the fast
    sup_antenna_names = np.setdiff1d(ant_pool_to_iterate, current_set_48)
    np.random.shuffle(sup_antenna_names)

    # random select 10 antennas from the fast and replace them with the antennas in the slow
    idx_to_be_replaced_in_cur = np.random.choice(len(current_set_48), 10, replace=False)
    idx_in_slow_chosen = np.random.choice(len(sup_antenna_names), 10, replace=False)
    replaced_antennas = [current_set_48[i] for i in idx_to_be_replaced_in_cur]
    new_in_antennas = [sup_antenna_names[i] for i in idx_in_slow_chosen]

    for i in range(10):
        current_set_48[idx_to_be_replaced_in_cur[i]] = sup_antenna_names[idx_in_slow_chosen[i]]

    new_set48_idx = np.array([slow_antenna_names.index(i) for i in current_set_48])
    replaced_idx = np.array([slow_antenna_names.index(i) for i in replaced_antennas])
    new_in_idx = np.array([slow_antenna_names.index(i) for i in new_in_antennas])



    # in slow data, flag all antennas except the fast antennas
    # get the flagging from np file
    flagging_src = np.load('/data07/peijinz/test-imaging/testdata/flagging.npy')

    # open the slow ms file
    tb = table()
    tb.open(fname_slow)
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flagging = flagging_src.copy()
    tb.close()

    # flag all antennas except the fast antennas

    for i in range(len(slow_antenna_names)):
        if slow_antenna_names[i] not in current_set_48:
            bad_ant1 = np.where(ant1 == i)[0]
            bad_ant2 = np.where(ant2 == i)[0]
            flagging[:,:,bad_ant1] = True
            flagging[:,:,bad_ant2] = True

    # write the flagging to the slow ms file
    tb.open(fname_slow, nomodify=False)
    tb.putcol('FLAG', flagging)
    tb.close()



    # do imaging
    odeconv.run_wsclean(fname_slow, imagename='flag_fast_ant', auto_mask=5, minuv_l='0', predict=False,
            size=512 , scale='2amin', pol='I', fast_vis=False, j=4,mem=5,beam_fitting_size=3)


    hdu2 = fits.open('flag_fast_ant-image.fits')
    res_img2 = np.concatenate([hdu2[0].data[0,0,0:100,0:100], hdu2[0].data[0,0,0:100,-100:], hdu2[0].data[0,0,-100:,0:100], hdu2[0].data[0,0,-100:,-100:]])
    snr = np.max(hdu2[0].data[0,0,:,:])/np.std(res_img2)

    str_res = compose_str_to_output(snr, old_set48_idx, new_set48_idx, replaced_idx, new_in_idx)
    print(str_res)

    # write the str_res to a text file
    with open('/data07/peijinz/test-imaging/testdata/ant_select_res.txt', 'a') as f:
        f.write(str_res+'\n')
    
    