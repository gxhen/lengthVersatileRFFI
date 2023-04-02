import numpy as np
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform, randn
import h5py
from scipy import signal
import math 
import random

def shuffle(data, label):

    index = np.arange(len(data))
    np.random.shuffle(index)
    data = data[index]
    label = label[index]

    return data, label


def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data 

def convert2complex(data):    
    num_row = data.shape[0]
    num_col = data.shape[1] 
    data_complex = np.zeros([num_row,round(num_col/2)],dtype=complex)
 
    data_complex = data[:,:round(num_col/2)] + 1j*data[:,round(num_col/2):]
 
    return data_complex


def load_single_file(filename, dataset_name, labelset_name, dev_range, pkt_range):
    f = h5py.File(filename,'r')
    label = f[labelset_name][:]
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1
    
    sample_index_list = []
    # label_selected = np.zeros([len(pkt_range)*len(dev_range),1])
    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)

    data = f[dataset_name][sample_index_list]
    data = convert2complex(data)
    
    label = label[sample_index_list]
      
    f.close()
    return data,label


def normalization(data):
    ''' Normalize the signal.'''
    s_norm = np.zeros(data.shape, dtype=complex)
    
    for i in range(data.shape[0]):
    
        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude**2))
        s_norm[i] = data[i]/rms
    
    return s_norm        
    
 
def spec_crop(x, crop_ratio):
     
    num_row = x.shape[0]
    x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

    return x_cropped
            

def ch_ind_spectrogram(data, win_len, crop_ratio):
    data = normalization(data)
    
    # win_len = 16
    overlap = round(0.5*win_len)
    
    num_sample = data.shape[0]
    # num_row = math.ceil(win_len*(1-2*crop_ratio))
    num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1) - 1
     
    
    data_dspec = np.zeros([num_sample, num_row, num_column, 1])
    # data_dspec = []
    for i in range(num_sample):
               
        dspec_amp = gen_ch_ind_spectrogram(data[i], win_len, overlap)
        dspec_amp = spec_crop(dspec_amp, crop_ratio)
        data_dspec[i,:,:,0] = dspec_amp
        # data_dspec[i,:,:,1] = dspec_phase
        
    return data_dspec   


def gen_ch_ind_spectrogram(sig, win_len, overlap):
    f, t, spec = signal.stft(sig, 
                            window='boxcar', 
                            nperseg= win_len, 
                            noverlap= overlap, 
                            nfft= win_len,
                            return_onesided=False, 
                            padded = False, 
                            boundary = None)
    
    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    # spec = spec_crop(spec, crop_ratio)
    
    # dspec = np.zeros([spec.shape[0],spec.shape[1]-1], dtype = complex)
    # for j in range(dspec.shape[1]):
    #     dspec[:,j] = spec[:,j] / spec[:,j+1]
        
    dspec = spec[:,1:]/spec[:,:-1]    
             
    dspec_amp = np.log10(np.abs(dspec)**2)
    # dspec_phase = np.angle(dspec)
              
    return dspec_amp


def data_generator(data_source, label_source, batch_size, snr_range, data_type = 'spatial'):
    """Generate a triplets generator for training."""
    
    num_data_block = len(data_source)
    
    while True:
        
        data_block_ind = random.randint(0,num_data_block - 1)
        
        data = data_source[data_block_ind]
        label = label_source[data_block_ind]
        
        
        sample_ind = random.sample(range(0, len(data)), batch_size)
        
        data = data[sample_ind]
        data = awgn(data, snr_range)
        data = ch_ind_spectrogram(data, win_len = 64, crop_ratio=0)
        
        if data_type == 'sequential':
            data = data[:,:,:,0]
            data = data.transpose(0,2,1) # [samples, timesteps, features]
        
        label = label[sample_ind]
        
        yield data, label  