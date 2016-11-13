import numpy as np
import numpy.lib.stride_tricks as npst  # for convolutional net
# import pdb
# import time
import sys  # for sys.float_info.epsilon


class LinearLayer:
    def __init__(self, nInputs, nUnits):
        self.nInputs = nInputs
        self.nUnits = nUnits
        self.firstLayer = False
        self.initWeights()

    def setFirstLayer(self, value=True):
        self.firstLayer = value

    def initWeights(self):
        scale = 1.0/np.sqrt(self.nInputs)
        # print('Wscale',scale)
        self.W = np.random.uniform(-scale, scale, size=(1+self.nInputs, self.nUnits))
        # print(self.W)
        
    def _forwardPass(self, X):
        self.X = X
        self.Y = np.dot(X, self.W[1:, :]) + self.W[0:1, :]
        return self.Y

    def _backwardPass(self, delta):
        # print("ll X.shape",self.X.shape,"delta.shape",delta.shape)
        self.dW = np.vstack((np.sum(delta,0), np.dot(self.X.T, delta)))
        # print("dW.shape",self.dW.shape)
        # print(delta.shape,self.W[1:,:].T.shape)
        if self.firstLayer:
            return None

        deltaPreviousLayer = np.dot(delta, self.W[1:,:].T)
        return deltaPreviousLayer

    def purge(self):
        """Release all matrices other than weights."""
        self.X = None
        self.Y = None

    def __repr__(self):
        return "LinearLayer({},{})".format(self.nInputs,self.nUnits)
    
    def __str__(self):
        return "LinearLayer({},{}) has W shape {}".format(self.nInputs,
                        self.nUnits,self.W.shape)

class TanhLayer(LinearLayer):
    # def __init__(self,nInputs,nUnits):
    #     super().__init__(nInputs,nUnits)

    def _forwardPass(self,X):
        super()._forwardPass(X)
        self.Y = np.tanh(self.Y)
        return self.Y

    def _backwardPass(self,delta):
        delta = delta * (1 - self.Y**2)
        return super()._backwardPass(delta)

    def __repr__(self):
        return "TanhLayer({},{})".format(self.nInputs,self.nUnits)
    
    def __str__(self):
        return "TanhLayer({},{}) has W shape {}".format(self.nInputs,
                        self.nUnits,self.W.shape)

class MultinomialLayer(LinearLayer):

    def __init__(self,nInputs,nUnits):
        super().__init__(nInputs,nUnits-1)

    def _forwardPass(self,X):
        super()._forwardPass(X)
        # Convert outputs to multinomial distribution
        mx = np.max(self.Y)
        expY = np.exp(self.Y-mx)
        denom = np.exp(-mx) + np.sum(expY,axis=1).reshape((-1,1)) + sys.float_info.epsilon
        self.Y = np.hstack((expY / denom, np.exp(-mx)/denom))
        return self.Y
        # rowsHavingZeroDenom = denom == 0.0
        # nZeroDenoms = np.sum(rowsHavingZeroDenom)
        # if nZeroDenoms > 0:
        #     Yzshape = (nZeroDenoms,expY.shape[1]+1)
        #     nClasses = Yzshape[1]
        #     # add random values to result in random choice of class
        #     Ytweaks = np.ones(Yzshape) * 1.0/nClasses + np.random.uniform(0,0.1,Yzshape)
        #     Ytweaks /= np.sum(Ytweaks,1).reshape((-1,1))
        #     Y = np.hstack((expY, np.tile(np.exp(-mx),(expY.shape[0],1))))
        #     zd = rowsHavingZeroDenom.squeeze()
        #     Y[zd,:] = Ytweaks # to choose class randomly
        #     nonzd = zd == False
        #     Y[nonzd,:] /= denom[nonzd,:]
        # else:
        #     Y = np.hstack((expY / denom, np.exp(-mx)/denom))
        # self.Y = Y
        # return Y
        
    def _backwardPass(self,delta):
        return super()._backwardPass(delta)

    def __repr__(self):
        return "MultinomialLayer({},{})".format(self.nInputs,self.nUnits)
    
    def __str__(self):
        return "MultinomialLayer({},{}) has W shape {}".format(self.nInputs,
                        self.nUnits,self.W.shape)

class MultinomialLayerOld(LinearLayer):

    def __init__(self,nInputs,nUnits):
        super().__init__(nInputs,nUnits-1)

    def _forwardPass(self,X):
        super()._forwardPass(X)
        # Convert outputs to multinomial distribution
        mx = np.max(self.Y)
        expY = np.exp(self.Y-mx)
        denom = np.exp(-mx) + np.sum(expY,axis=1).reshape((-1,1))
        rowsHavingZeroDenom = denom == 0.0
        nZeroDenoms = np.sum(rowsHavingZeroDenom)
        if nZeroDenoms > 0:
            Yshape = (nZeroDenoms,expY.shape[1]+1)
            nClasses = Yshape[1]
            # add random values to result in random choice of class
            Y = np.ones(Yshape) * 1.0/nClasses + np.random.uniform(0,0.1,Yshape)
            Y /= np.sum(Y,1).reshape((-1,1))
        else:
            Y = np.hstack((expY / denom, np.exp(-mx)/denom))
        self.Y = Y
        return Y
        
    def _backwardPass(self,delta):
        return super()._backwardPass(delta)

    def __repr__(self):
        return "MultinomialLayer({},{})".format(self.nInputs,self.nUnits)
    
    def __str__(self):
        return "MultinomialLayer({},{}) has W shape {}".format(self.nInputs,
                        self.nUnits,self.W.shape)


class ConvolutionalLayer(TanhLayer):

    def __init__(self,inputSizes, windowSizes, windowStrides, nUnits, firstLayer = False):
        #inputSizes is n1 x n2 x n3 x nu  where all but last are dimensions to convolve over. Last is channels.
        if len(inputSizes)-1 != len(windowSizes) or len(windowSizes) != len(windowStrides):
            print("ConvolutionalLayer: inputSizes-1, windowSizes, and windowStrides must have same length.")
            return
        # self.nSamples = nSamples
        self.inputSizes = np.array(inputSizes)
        self.windowSizes = np.array(windowSizes)
        self.windowStrides = np.array(windowStrides)
        self.nUnits = nUnits
        # self.nWindows = np.array([int(ni/stridei) - windowi + 1
        #                  for (ni,windowi,stridei) in zip(self.inputSizes[:-1],self.windowSizes,self.windowStrides)])
        self.nWindows = np.array([max(1,int((ni - windowi + 1) / stridei))
                         for (ni,windowi,stridei) in zip(self.inputSizes[:-1],self.windowSizes,self.windowStrides)])
        # self.nWindows[self.nWindows==0] = 1
        # print(self.nWindows)

        # self.nextLayerConvolutional = False  # Used in _backwardPass. Change with call to setNextLayerConvolutional
        nInputs = np.prod(self.windowSizes) * self.inputSizes[-1]
        self.updateConvolution = True  # See _forwardPass
        self.firstLayer = firstLayer
        super().__init__(nInputs,nUnits)

    # def setNextLayerConvolutional(self):
    #     self.nextLayerConvolutional = True
        
    def _windowize(self,X, nSamples, inputSizes, nWindows, windowSizes, windowStrides):
        # inputSizes is n1 x n2 x n3 x nu, all but last are dimensions to convolve over.
        # nWindows is number of resulting windows for each convolution dimension
        # windowSizes is size of mask for each convolution dimension
        # windowStrides is strides for each convolution dimension
        # X.size must be nSamples x prod(inputSizes)

        deb = False
        
        if deb: print("WINDOWIZE:")
        if deb: print(" X.shape",X.shape)
        if deb: print(" nSamples",nSamples)
        if deb: print(" inputSizes",inputSizes)
        if deb: print(" nwindows",nWindows)
        if deb: print(" windowSizes",windowSizes)
        if deb: print(" windowStrides",windowStrides)

        # nSamples = X.shape[0]
        if nSamples != self.nSamples:
            raise ValueError('X.shape[0] is not equal to self.nSamples')
        if X.size != nSamples*np.prod(inputSizes):
            raise ValueError('_windowize: X.shape=' + str(X.shape) +'. X.size=' + str(X.size) + ' is not equal to nSamples*prod(inputSizes)=' + str(nSamples*np.prod(inputSizes)) + ' of inputSizes=' + str(inputSizes))
            # raise ValueError('X.shape=' + str(X.shape) + ' [1]='+str(X.shape[1]) + ' is not equal to prod=' + str(np.prod(inputSizes)) + ' of inputSizes=' + str(inputSizes))
        uPrev = inputSizes[-1]
        # inputSizes = inputSizes[:-1] # remove uPref
        # Make ( d1*d2*d3*u, d2*d3*u, d3*u, u)
        inputSizesRevCumProd = np.cumprod(inputSizes[::-1])[::-1]
        # if deb: print('inputSizeRCP',inputSizesRevCumProd)
        itemsize = X.itemsize

        if deb: print("X.shape",X.shape)
        if deb: print("shape",[nSamples] + nWindows.tolist() + windowSizes.tolist() + [uPrev])
        # if deb: print("strides",[inputSizesRevCumProd[0] * itemsize] +
        #       (windowStrides * inputSizesRevCumProd[1:] * itemsize).tolist() +
        #       (inputSizesRevCumProd[1:] * itemsize).tolist() +
        #       [itemsize])
        if deb: print("strides",[inputSizesRevCumProd[0] ] +
              (windowStrides * inputSizesRevCumProd[1:]).tolist() +
              (inputSizesRevCumProd[1:]).tolist() +
              [1])
        
        # a = time.time()
        Xw = npst.as_strided(X,
                shape= [nSamples] + nWindows.tolist() + windowSizes.tolist() + [uPrev],
                    strides = [inputSizesRevCumProd[0] * itemsize] +
                             (windowStrides * inputSizesRevCumProd[1:] * itemsize).tolist() +
                             (inputSizesRevCumProd[1:] * itemsize).tolist() +
                             [itemsize])
        # strid = time.time() - a
        # This next reshape takes a lot of time, because Xw, from as_strided
        # is not contiguous, so a new array must be made.
        # a = time.time()
        # resh = time.time() - a
        # print('stride time',strid,'reshape time',resh)
        return Xw.reshape((nSamples, np.prod(nWindows), np.prod(windowSizes) * uPrev))
                         
    def _forwardPass(self, X):
        self.nSamples = X.shape[0]
        # Avoid the time-consuming windowize if possible
        # if self.firstLayer and self.updateConvolution:
        if not self.firstLayer or self.updateConvolution:
            # print('Recalculating windows')
            Xw = self._windowize(X,self.nSamples, self.inputSizes, self.nWindows, self.windowSizes, self.windowStrides)
            self.Xw = Xw
            if self.firstLayer:
                self.updateConvolution = False
        else:
            Xw = self.Xw
        # print("_forwardPass: X.shape",X.shape,"Xw.shape",self.Xw.shape,"W.shape",self.W.shape)
        self.Yw = super()._forwardPass(Xw)
        self.Y = self.Yw.reshape((self.nSamples,-1))
        # keep self.Yw for backpropping
        return self.Y

    def purge(self):
        """Release all matrices other than weights."""
        self.Xw = None
        self.Yw = None
        self.X = None
        self.Y = None
        
    def _info(self):
        print('nSamples:',self.nSamples)
        print('inputSizes',self.inputSizes)
        print('windowSizes',self.windowSizes)
        print('windowStrides',self.windowStrides)
        print('nWindows',self.nWindows)
        print('Xw.shape',self.Xw.shape)
        print('Y.shape',self.Y.shape)
        
    def _backwardPass(self,delta):
        deb = False
        nu = self.nUnits
        if deb: print("Xw.shape",self.Xw.shape)
        nw = self.Xw.shape[-1] # components in a window
        if deb: print("Xw reshaped.T is",self.Xw.reshape((-1,nw)).T.shape)
        if deb: print("delta reshaped is", delta.reshape((-1,nu)).shape)
        if deb: print("W shape is",self.W.shape)
        delta = delta * (1 - self.Y**2)
        self.dW = np.vstack((delta.reshape((-1,nu)).sum(0),
            np.dot(self.Xw.reshape((-1,nw)).T,delta.reshape((-1,nu)))))
        if deb: print("dW shape is",self.W.shape)
        # Xw is n x (d1d2...) x ... x (w1w2...u)
        # delta is n x (d1d2...u)
        # Y is n x (d1d2...u)
        # Yw is n x (d1d2 ...) x u
        # W is (1+w1w2..wn) x u
        # ERROR. delta is nxd1d2.. x u when NEXT LAYER IS CONV LAYER

        # Rest of this not needed if this layer is the first layer!!
        # TO-DO
        # if not self.firstLayer:

        if self.firstLayer:
            return None
        
        if deb:  print('delta.shape',delta.shape)
        uPrev = self.inputSizes[-1]
        # Build version of delta padded on edges and interspersed with zeros
        nZerosEdge = self.windowSizes - 1
        nZerosBetween = self.windowStrides - 1
        # nZerosBetween = (self.windowStrides - 1) * (self.windowSizes - 1)
        nZeros = nZerosEdge * 2 + nZerosBetween * (self.nWindows - 1)
        if deb: print('nZerosEdge:',nZerosEdge)
        if deb: print('nZerosBetween:',nZerosBetween)
        if deb: print('nZeros:',nZeros)
        deltaPadded = np.zeros(([self.nSamples] + (self.nWindows + nZeros + (self.windowSizes-1)).tolist() + [self.nUnits]))
        if deb: print('deltaPadded.shape:',deltaPadded.shape)
        
        # copy delta into correct positions of zero array
        # For 3x3 window with stride of 1 and 10x10 input grid of 2 channels,
        # want slices to be [:, 2:4
        # Humm...may need a boolean mask to do this.  Nah.  slower.
        # For each conv dimension, want [0:windowsize]+ *dup nWindows
        slices = [slice(None)] + [slice(a,b,s) for a,b,s in
            zip(nZerosEdge,
                nZerosEdge + self.nWindows * (nZerosBetween+1),
                nZerosBetween+1)] + [slice(None)]
                # self.windowStrides + nZerosBetween)] + [slice(None)]
        if deb: print('slices:',slices)
        if deb: print('deltaPadded[slices].shape:',deltaPadded[slices].shape)
        delta = delta.reshape([self.nSamples] + self.nWindows.tolist() + [self.nUnits])
        if deb: print('delta.reshaped to ',delta.shape)
        deltaPadded[slices] = delta
        # Extract windows from deltaPadded to backpropagate deltas into virtual units in this layer
        # deltaPadded = np.arange(deltaPadded.size).reshape(deltaPadded.shape)

        deltaPaddedw = self._windowize(deltaPadded,self.nSamples,
                                       list(deltaPadded.shape[1:-1]) + [self.nUnits],
                                       self.inputSizes[:-1],
                                       self.windowSizes, np.ones(len(self.windowSizes))) #self.windowStrides)

        if deb: print('delta (removed last dimension, with size 1')
        if deb: print(delta[:,:,:,0])
        if deb: print('deltaPadded without last dim')
        #if deb: print((deltaPadded[:,:,:,0]!=0).astype(int))
        if deb: print(deltaPadded[:,:,:,0])
        if deb: print("windowize(dp,",deltaPadded.shape,self.nSamples,deltaPadded.shape[1:],self.inputSizes[:-1],self.windowSizes,self.windowStrides)
        if deb: print('deltaPaddedw.shape',deltaPaddedw.shape)
        if deb: print('deltaPaddedw')
        x = (deltaPaddedw[:,:,:]!=0).astype(int)
        #if deb: print(x.reshape((-1,5,5)))
        if deb: print(deltaPaddedw.reshape((-1,81,2,2)))
        # deltaPaddedw is ... nSamples x ??? XXX
        # flip convolution dimensions of W and swapaxes so that W becomes d1xd2xd3...x unitsPrev x units
        u = self.nUnits
        nConvDims = len(self.inputSizes)-1
        if deb: print('nConvDims:',nConvDims)
        # for slice example like this see
        #   http://stackoverflow.com/questions/13240117/reverse-an-arbitrary-dimension-in-an-ndarray
        if deb: print('W.shape:',self.W.shape)
        # is this next statement right?  Only if all window values for prevunit0 are before all for prevunit1 etc.
        # Wflipped = self.W[1:,:].T.reshape((self.windowSizes.tolist() + [uPrev,u]))[[slice(None,None,-1)]*nConvDims + [Ellipsis]]
        Wflipped = self.W[1:,:].T.reshape(([u] + self.windowSizes.tolist() + [uPrev]))[[slice(None)]+[slice(None,None,-1)]*nConvDims + [slice(None)]]
        # Now Wflipped is nUnits x d1 x d2 ... x nPrevUnits, with d1... flipped
        if deb: print('Wflipped.shape:',Wflipped.shape)
        # Now Wflipped is nUnits d1d2... x nPrevUnits
        #Wflipped = Wflipped.swapaxes(-1,-2) # were ...,nUnitsPrev,nUnits.  now are ...,nUnits,nUnitsPrev
        #if deb: print('Wflipped.swapaxes(-1,-2).shape:',Wflipped.shape)
        # Reshape both deltaPaddedw and Wflipped so dot can do the backprop
        # nwu does not count the bias weight, which is not in Wflipped because not needed for backprop
        nwu = np.prod(self.windowSizes) * self.nUnits 
        if deb: print('nwu (prod(windowSizes)*nUnits) :',nwu)
        deltaPaddedw = deltaPaddedw.reshape((-1,nwu))
        if deb: print('deltaPaddedw reshaped:',deltaPaddedw.shape)
        Wflipped = Wflipped.reshape((nwu,-1))
        if deb: print('Wflipped reshaped:',Wflipped.shape)
        deltaPreviousLayer = np.dot(deltaPaddedw, Wflipped)
        if deb: print('deltaPreviousLayer.shape:',deltaPreviousLayer.shape)

        # deltaPreviousLayer should be nSamples x d1 x d2 x ... x uPrev
        # ERROR. but it is nSamples d1d2... x uPrev.
        # Want it to be nSamples x d1d2...uPrev
        nSamples = self.Y.shape[0]
        if deb: print('deltaPreviousLayer reshaped:',deltaPreviousLayer.reshape((nSamples,-1)).shape)
        return deltaPreviousLayer.reshape((nSamples,-1))

    def __repr__(self):
        return "ConvolutionalLayer({},{},{},{})".format(self.inputSizes,self.windowSizes,self.windowStrides,self.nUnits)

    def __str__(self):
        return "ConvolutionalLayer({},{},{}) has {} units and W shape {}".format(
                    self.inputSizes,self.windowSizes,self.windowStrides,self.nUnits,self.W.shape)

# if __name__ == "__main__":
if False:

        ll = LinearLayer(10,3)
        print(ll)
        print("forwardPass = ",ll._forwardPass(np.arange(2*10).reshape((2,-1))))
        delta = np.array([[1,-1,1],[-1,1,-1]])
        print("backwardPass = ",ll._backwardPass(delta))

        tl = TanhLayer(10,3)
        print(tl)
        print("forwardPass = ",tl._forwardPass(np.arange(2*10).reshape((2,-1))))
        delta = np.array([[1,-1,1],[-1,1,-1]])
        print("backwardPass = ",tl._backwardPass(delta))

        ml = MultinomialLayer(10,3)
        print(ml)
        print("forwardPass = ",ml._forwardPass(np.arange(2*10).reshape((2,-1))))
        delta = np.array([[1,0],[0,1]])
        print("backwardPass = ",ml._backwardPass(delta))

        X = np.arange(2*100).reshape((2,10,10))
        cl = ConvolutionalLayer([10,10,1],[3,3],[1,1],2)
        print(cl)
        print("forwardPass shape = ",cl._forwardPass(X).shape)
        delta = np.arange(2*np.prod(cl.nWindows)*cl.nUnits).reshape((2,-1))
        print("backwardPass = ",cl._backwardPass(delta).shape)
