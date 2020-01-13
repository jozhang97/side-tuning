import torch
import numpy as np

class MedianImageMeter(object):
    def __init__(self, bit_depth, im_shape, device='cpu'):
        self.bit_depth = bit_depth
        self.im_shape = list(im_shape)
        self.device = device
        if bit_depth == 8:
            self.dtype = np.uint8
        elif bit_depth == 16:
            self.dtype = np.uint16
        else:
            raise NotImplementedError(
                "MedianMeter cannot find the median of non 8/16 bit-depth images.")
        self.reset()
    
    def reset(self):
        self.freqs = self.make_freqs_array()

    def add(self, val, mask=1):
        self.val = torch.LongTensor( val.astype(np.int64).flatten()[np.newaxis,:] ).to(self.device)

        if type(mask) == int:
            mask = torch.IntTensor(self.val.size()).fill_(mask).to(self.device)
        else:
            mask = torch.IntTensor(mask.astype(np.int32).flatten()[np.newaxis,:]).to(self.device)

        self.freqs.scatter_add_(0, self.val, mask) 
        self.saved_val = val

    def value(self):
        self._avg = np.cumsum(
                        self.freqs.cpu().numpy(), 
                        axis=0)
        self._avg = np.apply_along_axis(
                        lambda a: a.searchsorted(a[-1] / 2.), 
                        axis=0, 
                        arr=self._avg)\
                    .reshape(tuple([-1] + self.im_shape))
        return np.squeeze(self._avg, 0)

    def make_freqs_array(self):
        # freqs has shape N_categories x W x H x N_channels 
        shape = tuple([2**self.bit_depth] + self.im_shape)
        freqs = torch.IntTensor(shape[0], int(np.prod(shape[1:]))).zero_()
        return freqs.to(self.device)
