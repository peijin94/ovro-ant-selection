
import numpy as np

def compose_str_to_output(snr, old_set, new_set, replaced, new_in, *args, **kwargs):
    return (str(snr)+','+ ';'.join([str(idx) for idx in old_set]) + 
                    ',' + ';'.join([str(idx) for idx in new_set]) + 
                    ',' + ';'.join([str(idx) for idx in replaced]) + 
                    ',' + ';'.join([str(idx) for idx in new_in])+ 
                    ','+ ';'.join([str(i) for i in args]) )

def convert_str_to_var(s):
    snr, old_set, new_set, replaced, new_in, info = s.split(',')
    return (float(snr), [int(i) for i in old_set.split(';')], [int(i) for i in new_set.split(';')], 
                        [int(i) for i in replaced.split(';')], [int(i) for i in new_in.split(';')],
                        info)

def corner_rms(img, cornerpix=100):
    img = img.squeeze()
    imgrms = np.concatenate([img[0:cornerpix,0:cornerpix], img[0:cornerpix,-cornerpix:], 
                             img[-cornerpix:,0:cornerpix], img[-cornerpix:,-cornerpix:]])
    return np.max(img)/np.std(imgrms)

def psf_peak_thresh_area(img, thresh=0.3):
    img = img.squeeze()
    peak = np.max(img)
    mask = img > 0.3*peak
    return np.sum(mask)

def max_bonding_box(img, thresh=0.3):
    img = img.squeeze()
    mask = img > 0.3*np.max(img)
    mask = np.where(mask)
    # return the length of the bonding box diagonal
    return np.sqrt((np.max(mask[0])-np.min(mask[0]))**2 + (np.max(mask[1])-np.min(mask[1]))**2)
