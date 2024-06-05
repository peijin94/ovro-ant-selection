import numpy as np
import matplotlib.pyplot as plt

from ovrolwasolar import deconvolve as odeconv
from ovrolwasolar import flagging as oflag
import copy
from casatools import msmetadata, ms, table, quanta, measures
from astropy.io import fits
from baseline_optm import *

# print with color
def compose_str_with_color(text,color):
    return "\033[1;{};40m{}\033[0m".format(color,text)

r_dot_sign = compose_str_with_color('●', 31)
g_dot_sign = compose_str_with_color('●', 32)

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

n_iter  = 500
num_replace_per_iter = 8
core_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(0, 251)]
exp_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(251, 352)]

core_weight = 1
exte_weight = 20

rms_score_prev = 0
peak03_score_prev = 1e5
maxbox_score_prev = 1e5
score_prev = 0
select_with_ant_score = True

antenna_score_book = np.ones(len(slow_antenna_names))

for idx in range(n_iter):

    old_set48_idx = np.array([slow_antenna_names.index(i) for i in current_set_48])
    # random select 10 antennas from the slow which is not in the fast, replace random 10 antennas in the fast
    sup_antenna_names = np.setdiff1d(ant_pool_to_iterate, current_set_48)
    np.random.shuffle(sup_antenna_names)

    # apply weighting
    weight_arr = np.zeros(len(sup_antenna_names))
    for i in range(len(sup_antenna_names)):
        if sup_antenna_names[i] in core_ant_name_list:
            weight_arr[i] = core_weight
        elif sup_antenna_names[i] in exp_ant_name_list:
            weight_arr[i] = exte_weight
        # good antennas is more likely to be selected
        weight_arr[i] = weight_arr[i] * antenna_score_book[slow_antenna_names.index(sup_antenna_names[i])]

    # random select 10 antennas from the fast and replace them with the antennas in the slow
    idx_to_be_replaced_in_cur = np.random.choice(len(current_set_48), num_replace_per_iter, replace=False)
    idx_in_slow_chosen = np.random.choice(len(sup_antenna_names), num_replace_per_iter, p=weight_arr/np.sum(weight_arr), replace=False)
    replaced_antennas = [current_set_48[i] for i in idx_to_be_replaced_in_cur]
    new_in_antennas = [sup_antenna_names[i] for i in idx_in_slow_chosen]

    if idx != 0:
        for i in range(num_replace_per_iter):
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
    odeconv.run_wsclean(fname_slow, imagename='flag_fast_ant', auto_mask=5, minuv_l='0', predict=False, rm_misc=False,
            size=512 , scale='1.5amin', pol='I', fast_vis=False, j=4,mem=5,beam_fitting_size=3)

    hdu2 = fits.open('flag_fast_ant-image.fits')
    
    dataimg = hdu2[0].data[0,0,:,:].squeeze()

    hdu_psf = fits.open('flag_fast_ant-psf.fits')
    psfimg = hdu_psf[0].data[0,0,:,:].squeeze()

    rms_score = corner_rms(dataimg) # larger is better
    peak03_score = psf_peak_thresh_area(psfimg) # smaller is better
    #maxbox_score = max_bonding_box(psfimg) # smaller is better
    maxbox_score = hdu2[0].header['BMAJ']*60 # smaller is better

    score= (rms_score/10 +  
            300/(peak03_score*(hdu2[0].header['CDELT1'])**2*3600) + 
            50/(maxbox_score))

    str_res = compose_str_to_output(score, old_set48_idx, new_set48_idx, replaced_idx, new_in_idx,
                            rms_score, peak03_score, maxbox_score)

    # write the str_res to a text file
    with open('/data07/peijinz/test-imaging/testdata/ant_select_res_2.txt', 'a') as f:
        f.write(str_res+'\n')


    if idx == 0:
        rms_score_base = rms_score
        peak03_score_base = peak03_score
        maxbox_score_base = maxbox_score
        score_base = score
        rms_score_prev = rms_score
        peak03_score_prev = peak03_score
        maxbox_score_prev = maxbox_score
        # print blue
        print( compose_str_with_color('[init]', 30 ) + '\t idx:' + str(idx).rjust(4," ") + '; score:' + str('%.2f' % score)
               + '; snr:' + str('%.2f' % rms_score) + '; bArea:' + str('%.2f' % (peak03_score*(hdu2[0].header['CDELT1'])**2 *3600)) + '; bmag' + str('%.2f' % maxbox_score))
    else:
        # if rms decrease, peak03 increase, maxbox increase, accept the change
        if ((  rms_score < rms_score_prev * 0.95 
            and peak03_score > peak03_score_prev / 0.9 
            and maxbox_score > maxbox_score_prev / 0.9) 
            or (score < 0.96 * score_prev)
            or rms_score < rms_score_prev * 0.8
            or peak03_score > peak03_score_prev/0.6
            or maxbox_score > maxbox_score_prev/0.6
            or rms_score < rms_score_base * 0.9
            or peak03_score > peak03_score_base/0.6
            or maxbox_score > maxbox_score_base/0.6
            or score < 0.9 * score_base
            ):
            # reject the change
            current_set_48 = [slow_antenna_names[i] for i in old_set48_idx]
            # format the float to %.2f int to %d


            # bad antenna apply deduction score
            if (rms_score < rms_score_prev and
                    peak03_score > peak03_score_prev and
                    maxbox_score > maxbox_score_prev and
                    score >0 and score_prev>0):
                antenna_score_book[replaced_idx] = antenna_score_book[replaced_idx] * score_prev/score
                antenna_score_book[new_in_idx] = antenna_score_book[new_in_idx] * score/score_prev
                score_updated = "\t ant score updated"
            else:
                score_updated = ""

            print( compose_str_with_color('[reject]', 31) + '\t idx:' + str(idx) + 
                     '\t score:' + str('%.2f' % score) + (r_dot_sign if score < score_prev else g_dot_sign)
                   + '\t snr:' + str('%.2f' % rms_score) + (r_dot_sign if rms_score < rms_score_prev  else g_dot_sign)
                   + '\t bArea:' + str('%.2f' % (peak03_score*(hdu2[0].header['CDELT1'])**2*3600)) + (r_dot_sign if peak03_score > peak03_score_prev else g_dot_sign) 
                   + '\t bmaj:' + str('%.2f' % maxbox_score) + (r_dot_sign if maxbox_score > maxbox_score_prev else g_dot_sign) 
                   + score_updated)
            
        else:
            
            # good antenna apply addition score
            if (rms_score > rms_score_prev*0.99 and
                    peak03_score < peak03_score_prev/0.99 and
                    maxbox_score < maxbox_score_prev/0.99 and
                    score_prev>0 and score>0):
                antenna_score_book[replaced_idx] = antenna_score_book[replaced_idx] * score_prev/score
                antenna_score_book[new_in_idx] = antenna_score_book[new_in_idx] * score/score_prev
                score_updated = "\t ant score updated"
            else:
                score_updated = ""
            
            print( compose_str_with_color('[accept]', 32) + '\t idx:' + str(idx) +
                    '\t score:' + str('%.2f' % score) + (r_dot_sign if score < score_prev else g_dot_sign)
                     + '\t snr:' + str('%.2f' % rms_score) + (r_dot_sign if rms_score < rms_score_prev  else g_dot_sign)
                     + '\t bArea:' + str('%.2f' % (peak03_score*(hdu2[0].header['CDELT1'])**2*3600)) + (r_dot_sign if peak03_score > peak03_score_prev else g_dot_sign) 
                     + '\t bmaj:' + str('%.2f' % maxbox_score) + (r_dot_sign if maxbox_score > maxbox_score_prev else g_dot_sign)
                     + score_updated)

            rms_score_prev = rms_score
            peak03_score_prev = peak03_score
            maxbox_score_prev = maxbox_score
            score_prev = score

# save score book to text file
np.savetxt('/data07/peijinz/test-imaging/testdata/antenna_score_book.txt', antenna_score_book)
            
            

    
    
    
    