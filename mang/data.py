import numpy as np
import cudamat.gnumpy as gnp
import scipy.io as sio

'''
A class that feeds training data into a deep network
 - the data is assumed to be pre-processed in prior to loading
 - this class does not support shuffling data indices
'''

class Data:
    def __init__(self, source, param=None):
        self.i_src = 0
        if type(source) == type(self):
            self.n_src = source.n_src
            self.data_loader = source.data_loader
            self.data_key = source.data_key
            if type(source.src) == np.ndarray:
                self.src = np.array(source.src)
            elif type(source.src) == list and type(source.src[0]) == str:
                self.src = [str(x) for x in source.src]
        elif type(source) == np.ndarray:
            self.n_src = 1
            self.src = np.array(source)
            self.data_loader = None
            self.data_key = None
        elif type(source) == list and type(source[0]) == str:
            self.n_src = len(source)
            self.src = [str(x) for x in source]
            if source[0][-4:] == '.npy':
                self.data_loader = np.load
                self.data_key = lambda x: np.array(x)
            elif source[0][-4:] == '.mat':
                self.data_loader = sio.loadmat
                self.data_key = param
            elif source[0][-7:] == '.pkl.gz':
                self.data_loader = U.load_pickle
                self.data_key = param
        else:
            raise

    def __getitem__(self, i_src):
        if self.i_src < self.n_src:
            if self.data_loader != None:
                tmp = self.data_loader(self.src[i_src])
                return self.data_key(tmp)
            else:
                return self.src
        else:
            raise IndexError

    def __len__(self):
        return self.n_src

    def next(self):
        if self.i_src < self.n_src:
            self.i_src += 1
            if self.data_loader != None:
                tmp = self.data_loader(self.src[i_src])
                return self.data_key(tmp)
            else:
                return self.src
        else:
            raise StopIteration

    def __iter__(self):
        for i_src in xrange(self.n_src):
            if self.data_loader != None:
                tmp = self.data_loader(self.src[i_src])
                yield self.data_key(tmp)
            else:
                yield self.src
        else:
            raise StopIteration

if __name__ == '__main__':
    import os
    print 'testing for single numpy array'
    data = sio.loadmat('/home/e0en/Codes/research/dataset/mnist.mat')['X']
    data = Data(data)
    x = data.next()
    print x.shape
    for x in data:
        pass
    print '...done!'

    print 'testing for data composed of multiple files'
    data_dir = '/home/e0en/Codes/research/dataset/flickr'
    image_dir = os.path.join(data_dir, 'image/unlabelled')
    image_files= [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
    image_files.sort()
    data = Data(image_files)
    for x in data:
        pass
    print '...done!'
