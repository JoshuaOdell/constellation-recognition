import scaledconjugategradient as scg
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import mlutils as ml
import random #for random.sample in epsilonGreedy
from copy import copy
#import numpy.lib.stride_tricks as npst  # for convolutional net
import pdb
import time

from layers import LinearLayer, TanhLayer, MultinomialLayer, ConvolutionalLayer

######################################################################

class NeuralNetworkByLayers:

    def __init__(self,layers,*otherargs):
        self.layers = layers
        self.Xmeans = self.Xstds = None
        self.Tmeans = self.Tstds = None
        self.iteration = 0 #mp.Value('i',0)
        self.numberOfIterations = 0
        self.trained = False #mp.Value('b',False)
        self.reason = None
        self.errorTrace = []
        self.frozenHiddenLayers = False
        self.layers[0].setFirstLayer(True)
        # Performed by layer constructors
        # self.initWeights()
        self.timings = None
        
    def initWeights(self):
        for layer in self.layers:
            layer.initWeights()
        
    def getErrorTrace(self):
        return self.errorTrace
    
    def getNumberOfIterations(self):
        return self.numberOfIterations

    def use(self,X,allOutputs=False):
        if self.Xmeans is not None and not isinstance(self.layers[0],ConvolutionalLayer):
            X = self._standardizeX(X)
        if isinstance(self.layers[0],ConvolutionalLayer):
            self.layers[0].updateConvolution = True
            print('use: layer 0 is convolutional and updateConvolution set to True')
        Y = self._forwardPass(X)
        if self.Tmeans is not None:
            Y = self._unstandardizeT(Y)
        # update last layer output with unstandardized output
        # Which means cannot use .use during optimization
        self.layers[-1].Y = Y  # to 
        return Y if not allOutputs else [layer.Y for layer in self.layers]

    def _forwardPass(self,Y):
        # timei = 0
        # a = time.time()
        for layer in self.layers:
            Y = layer._forwardPass(Y)
            # b = time.time()
            # self.timings.append(['fp',timei,b-a])
            # timei += 1
            # a = b
        return Y

    def _backwardPass(self,delta):
        # timei = 0
        # a = time.time()
        for layer in reversed(self.layers):
            delta = layer._backwardPass(delta) # last delta not needed
            # b = time.time()
            # self.timings.append(['bp',timei,b-a])
            # a = b
            # timei += 1
        # return [layer.dW for layer in self.layers]
    
    def _preProcessTargets(self,T):
        if self.Tmeans is None and T.shape[1] > 1:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self._standardizeT(T)
        return T

    def _objectiveF(self,w, X,T):
            self._unpack(w)
            Y = self._forwardPass(X)
            return 0.5 * np.mean((Y - T)**2)
        
    def _gradF(self,w, X,T):
        self._unpack(w)
        Y = self._forwardPass(X)
        delta = (Y - T) / (X.shape[0] * T.shape[1])
        self._backwardPass(delta)  
        return self._pack([layer.dW for layer in self.layers])

    def _scg_verbose_eval(self):
        return lambda objValue: np.sqrt(objValue)

    # Call setInputRanges to not rely on train to calculate good means and
    # stds from first minibatch
    def setInputRanges(self,inputMinsMaxs):
        minsmaxs = np.array(inputMinsMaxs)
        self.Xmeans = minsmaxs.mean(axis=1)
        self.Xstds = minsmaxs.std(axis=1)
        self.Xconstant = self.Xstds == 0 # should all be false
        # print('Xstds and Xconstant',self.Xstds, self.Xconstant)
        self.XstdsFixed = self.Xstds
        
    def train(self,X,T, nIterations=100,
              weightPrecision=0,errorPrecision=0,verbose=False):
        self.timings = []
        if not isinstance(self.layers[0],ConvolutionalLayer):
            if X.shape[1] != self.layers[0].nInputs:
                print('Number of columns in X plus 1 ({}) does not equal the number of declared network inputs ({}).'.format(X.shape[1],self.layers[0].nInputs))
                return #sys.exit(1)
            if self.Xmeans is None:
                self.Xmeans = X.mean(axis=0)
                self.Xstds = X.std(axis=0)
                # self.Xconstant = (self.Xstds == 0).reshape((1,-1))
                self.Xconstant = self.Xstds == 0
                self.XstdsFixed = copy(self.Xstds)
                self.XstdsFixed[self.Xconstant] = 1
            X = self._standardizeX(X)
        else:
            pass #print("TO-DO: Add standardization code to ConvolutionalLayer")

        T = self._preProcessTargets(T)

        if isinstance(self.layers[0],ConvolutionalLayer):
            self.layers[0].updateConvolution = True
            
        scgresult = scg.scg(self._pack([layer.W for layer in self.layers]),
                            self._objectiveF, self._gradF, X,T,
                            evalFunc=self._scg_verbose_eval(),
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace += scgresult['ftrace'].tolist()
        self.numberOfIterations = len(self.errorTrace) - 1
        # self.trained.value = True
        self.trained = True
        return self

    def _standardizeX(self,X):
        if self.Xmeans is not None:
            result = (X - self.Xmeans) / self.XstdsFixed
            result[:,self.Xconstant] = 0.0
        else:
            result = X
        return result
    def _unstandardizeX(self,Xs):
        if self.Xmeans is not None:
            result = self.Xstds * Xs + self.Xmeans
        else:
            result = Xs
        return result
    def _standardizeT(self,T):
        if self.Tmeans is not None:
            result = (T - self.Tmeans) / self.TstdsFixed
            result[:,self.Tconstant] = 0.0
        else:
            result = T
        return result
    def _unstandardizeT(self,Ts):
        if self.Tmeans is not None:
            result = self.Tstds * Ts + self.Tmeans
        else:
            result = Ts
        return result
    def _pack(self,WsOrdWs):
        return np.hstack([w.flat for w in WsOrdWs])
    def _unpack(self,w):
        first = 0
        for layer in self.layers:
            last = first + layer.W.size - 1
            layer.W[:] = w[first:last+1].reshape((layer.W.shape))
            first = last + 1

    def draw(self,inputNames = None, outputNames = None):
        weightMatrices = [layer.W for layer in self.layers]
        ml.draw(weightMatrices, inputNames, outputNames)

    def __repr__(self):
        str = 'NeuralNetwork(' + ',\n              '.join(['{!r}'.format(layer) for layer in self.layers])
        # {})'.format(self.layers)
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str

    def __str__(self):
        return self.__repr__()
    
######################################################################

class NeuralNetwork(NeuralNetworkByLayers):
    def __init__(self,nUnits,*otherargs):
        layers,ni = self._createAllButLastLayer(nUnits,*otherargs)
        layers.append(self._createFinalLayer(ni,nUnits[-1]))
        super().__init__(layers)

    def _createAllButLastLayer(self,nUnits): # *args):
        layers = []
        ni = nUnits[0]
        for nu in nUnits[1:-1]:
            layers.append(TanhLayer(ni,nu))
            ni = nu
        return layers,ni

    def _createFinalLayer(self,ni,nu):
        return LinearLayer(ni,nu)

######################################################################

class NeuralNetworkClassifier(NeuralNetwork):  # ByLayers):
    def __init__(self, nUnits, *otherargs):
        super().__init__(nUnits, *otherargs)

    def _createFinalLayer(self,ni,nu):
        return MultinomialLayer(ni,nu)

    def _preProcessTargets(self,T):
        self.classes,counts = np.unique(T,return_counts=True)
        if self.layers[-1].nUnits != len(self.classes)-1:
            raise ValueError(" In NeuralNetworkClassifier, the number of outputs must be one less than\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetworkClassifier()." % (self.nus[-1], len(self.classes)))
        self.mostCommonClass = self.classes[np.argmax(counts)]  # to break ties, in use
        # make indicator variables for target classes
        Ti = (T == self.classes).astype(int)
        return Ti

    def _objectiveF(self,w, X,T):
            self._unpack(w)
            # if isinstance(self.layers[0],ConvolutionalLayer):
            #     print('_objectiveF, updateConv is',self.layers[0].updateConvolution)
            Y = self._forwardPass(X)
            return -np.mean(T * np.log(Y))

    def _gradF(self,w, X,T):
            self._unpack(w)
            Y = self._forwardPass(X)
            delta = (Y[:,:-1] - T[:,:-1]) / (X.shape[0] * (T.shape[1]-1))
            self._backwardPass(delta)  
            return self._pack([layer.dW for layer in self.layers])

    def _scg_verbose_eval(self):
        return lambda objValue: np.exp(-objValue)

    def use(self,X,allOutputs=False):
        if not isinstance(self.layers[0],ConvolutionalLayer):
            X = self._standardizeX(X)
        else:
            self.layers[0].updateConvolution = True
        Y = self._forwardPass(X)
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        # If any row has all equal values, then all classes have same probability.
        # Let's return the most common class in these cases
        classProbsEqual = (Y == Y[:,0:1]).all(axis=1)
        if sum(classProbsEqual) > 0:
            classes[classProbsEqual] = self.mostCommonClass
        return classes if not allOutputs else [classes, self.layers[-1].Y, [layer.Y for layer in self.layers[:-1]]]

######################################################################


class NeuralNetworkConvolutional(NeuralNetwork):

    def __init__(self,nUnits,inputSize,windowSizes=((3,3),(5,5)),windowStrides=((1,1),(2,2))):
        super().__init__(nUnits,inputSize,windowSizes,windowStrides)
        print("ToDo! Implement standardization of input for convolutional net.")

    def _createAllButLastLayer(self,nUnits,*otherargs):
        inputSize,windowSizes,windowStrides = otherargs
        if len(windowSizes) != len(windowStrides):
            print("NeuralNetworkConvolutional: ERROR. len(windowSizes) != len(windowStrides)")
            return
        # check number of dimensions to convolve over
        allSizes = [len(inputSize)] + [len(a) for a in windowSizes] + [len(a) for a in windowStrides]
        if allSizes[1:] != allSizes[:-1]:
            print("NeuralNetworkConvolutional: ERROR. len(inputSize) and length of each windowSizes and windowStrides are not equal.")
            return
        nLayers = len(nUnits)-1
        nConvLayers = len(windowSizes)
        if nLayers < nConvLayers:
            print("NeuralNetworkConvolutional: ERROR. len(nUnits)-1 not greater than or equal to number of convolutional layers.")
            return
            
        if nConvLayers > 0:
            layers = [ConvolutionalLayer(list(inputSize) + [nUnits[0]],
                                         windowSizes[0],
                                         windowStrides[0],
                                         nUnits[1])]
            for layeri in range(1,nConvLayers):
                layers.append(ConvolutionalLayer(layers[-1].nWindows.tolist() + [nUnits[layeri]],
                                                 windowSizes[layeri],
                                                 windowStrides[layeri],
                                                 nUnits[layeri+1]))
            nInputsNextLayer = np.prod(layers[-1].nWindows) * layers[-1].nUnits
        else:
            nInputsNextLayer = nUnits[0]

        for layeri in range(nConvLayers,nLayers-1):
            layers.append(TanhLayer(nInputsNextLayer,nUnits[layeri+1]))
            nInputsNextLayer = nUnits[layeri+1]
        
        return layers, nInputsNextLayer

# Multiple inheritance explained: http://www.python-course.eu/python3_multiple_inheritance.php

######################################################################


class NeuralNetworkConvolutionalClassifier(NeuralNetworkClassifier):

    def __init__(self, nUnits, inputSize, windowSizes=((3, 3), (5, 5)), windowStrides=((1, 1), (2, 2))):
        super().__init__(nUnits,inputSize,windowSizes,windowStrides)

    def _createAllButLastLayer(self,nUnits,*otherargs):
        return NeuralNetworkConvolutional._createAllButLastLayer(self,nUnits,*otherargs)

if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    plt.ion()
    # plt.close('all')

    if True:
        n = 10
        X = np.arange(n).reshape((-1,1))
        T = X + np.random.randn(n,1)

        # Fully connected net mapping samples of 100 input variables to 10 output variables,
        # using 3 hidden layers of 20,5,20 units.
        usualFFRegression = NeuralNetwork([1, 20, 5, 20, 1])
        startTime = time.time()
        usualFFRegression.train(X,T,100)
        y = usualFFRegression.use(X)
        print("took {} seconds".format(time.time()-startTime))
        result = np.hstack((T,y))
        plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(usualFFRegression.getErrorTrace())
        plt.subplot(2,1,2)
        plt.plot(result)

        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        T = np.array([[0],  [1],  [1],  [0]])
        # Fully connected net classifying 4 2-dimensional samples (binary valued components)
        # according to the exclusive-or of the two inputs, using 2 hidden layers of 5 and 4 units.
        usualFFClassifier = NeuralNetworkClassifier([2,5,4,2])
        startTime = time.time()
        usualFFClassifier.train(X,T,100)
        y = usualFFClassifier.use(X)
        print("took {} seconds".format(time.time()-startTime))
        result = np.hstack((T,y))
        plt.figure(2)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(usualFFClassifier.getErrorTrace())
        plt.subplot(2,1,2)
        plt.plot(result,'o-')
        plt.ylim(-0.1,1.1)



    # Mapping 256*10 samples over time of 8-channel EEG to single output (velocity of arm),
    # using one convolutional layer of 10 units with 20-component window over the one time dimension with stride of 1,
    # then a second convolutional layer of 30 units with 200-component window over the one time dimension with stride of 25,
    # then a fully connected layer of 10 units,
    # then an output layer of 1 unit.
    # convRegression = NeuralNetworkConvolutional([8,10,30,1], [2560], [[20],[200]], [[1],[25]])

        
    # Classifying 20x20 r,g,b images using one convolutional layer of 5 units with 3x3 windows and 1x1 strides,
    # then a second convolutional layer of 8 units with 4x4 windows and 2x2 strides,
    # then a fully connected layer of 10 units,
    # then an output layer of 2 units for the two classes.
    # convClassifier = NeuralNetworkConvolutionalClassifier([3,10,5,3,2] ,[20,20], [[5,5], [5,5]], [[1,1],[3,3]])



    print('Making image data')

    # Test convClassifier
    # Make 20x20 black and white images with diamonds or squares for the two classes, as line drawings.
    def makeImages(nEach):
        images = np.zeros((nEach*2, 20,20,1))
        radii = 3 + np.random.randint(10-5,size=(nEach*2,1))
        centers = np.zeros((nEach*2,2))
        for i in range(nEach*2):
            r = radii[i,0]
            centers[i,:] = r +1 + np.random.randint(18 - 2*r, size=(1,2))
            x = int(centers[i,0])
            y = int(centers[i,1])
            if i < nEach:
                # squares
                images[i,x-r:x+r,y+r,0] = 1.0
                images[i,x-r:x+r,y-r,0] = 1.0
                images[i,x-r,y-r:y+r,0] = 1.0
                images[i,x+r,y-r:y+r+1,0] = 1.0
            else:
                # diamonds
                images[i,range(x-r,x),range(y,y+r),0] = 1.0
                images[i,range(x-r,x),range(y,y-r,-1),0] = 1.0
                images[i,range(x,x+r+1),range(y+r,y-1,-1),0] = 1.0
                images[i,range(x,x+r),range(y-r,y),0] = 1.0
        # images += np.random.randn(*images.shape) * 0.5
        T = np.ones((nEach*2,1))
        T[nEach:] = 2
        return images,T

    nEach = 100
    X,T = makeImages(nEach)
    # X = X[0:2,:10,:10,:]
    # T = T[0:2,:]
    # T[1,:] = 2
    
    # Fully connected net classifying 20x20 (=400 flattened) images with r,g,b channels into two classes, using 2 hidden layers of 100 and 100 units.
    usualFFClassifier = NeuralNetworkClassifier([20*20, 100, 100, 2])
    a = time.time()
    Xflat = X.reshape((200,-1))
    usualFFClassifier.train(Xflat,T,100)
    Pc = usualFFClassifier.use(Xflat)
    print(usualFFClassifier)
    print("took {} seconds".format(time.time()-a))
    # print(np.hstack((T,Pc)))
    print("fraction training correct",sum(T==Pc) / T.shape[0])
    plt.figure(3)
    plt.clf()
    plt.plot(usualFFClassifier.getErrorTrace())


    if False:
        plt.figure(4)
        plt.clf()
        mx = int(np.sqrt(nEach*2)) + 1
        for i in range(nEach*2):
            plt.subplot(mx,mx,i+1)
            plt.imshow(X[i,:,:,0],interpolation='nearest',cmap=plt.cm.binary)
            plt.axis('off')
            # plt.title("Class {}".format(T[i,0]))

    # convClassifier = NeuralNetworkConvolutionalClassifier(
    #     nUnits = [1,5,4,10,2],
    #     inputSize = [20,20],
    #     windowSizes = [[3,3],[15,15]],
    #     windowStrides = [[1,1],[1,1]])
    print('Making convClassifier')

    if True:
        convClassifier = NeuralNetworkConvolutionalClassifier(
            nUnits = [1,5,6,4,2],
            #inputSize = [20,20],
            inputSize = [20,20],
            windowSizes = [[3,3],[8,8]],
            windowStrides = [[1,1],[5,5]])
    else:
        convClassifier = NeuralNetworkConvolutionalClassifier(
            nUnits = [1,1,1,10,2],
            inputSize = [20,20],
            windowSizes = [[15,15],[2,2]],
            windowStrides = [[1,1],[1,1]])

    # windowStrides = [[2,2]])  ERROR

    print(convClassifier)
    a = time.time() 
    print('Training convClassifier')
    convClassifier.train(X,T,500,verbose=True)
    Pc = convClassifier.use(X)
    print("took {} seconds".format(time.time()-a))
    print(np.hstack((T,Pc)).T)
    print("Train fraction correct",sum(T==Pc) / T.shape[0])

    Xtest,Ttest = makeImages(10)
    # Xtest = Xtest[:,:10,:10,:]
    Pctest = convClassifier.use(Xtest)
    # print(np.hstack((Ttest,Pctest)).T)
    print("Test fraction correct",sum(Ttest==Pctest) / Ttest.shape[0])
    
    allOut = convClassifier.use(Xtest,allOutputs=True)
    plt.figure(5);
    plt.clf()
    plt.plot(allOut[1],'o-')
    nTest = Xtest.shape[0]
    for i in range(nTest):
        w = 1/(nTest+7)
        ax = plt.axes([0.12+i*w, 0.5-w, w,w])
        # plt.subplot(4,5,i+1)
        plt.imshow(Xtest[i,:,:,0],interpolation='nearest',cmap=plt.cm.binary)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        # plt.axis('off')
        # plt.title("Class {}".format(T[i,0]))

    
    # Look at filters
    plt.figure(6)
    plt.clf()
    # plt.subplot(1,2,1)
    lay = convClassifier.layers[0]
    wmagmax = np.max(np.abs(lay.W[1:,:]))
    for i in range(lay.nUnits):
        plt.subplot(4,4,i+1)
        ml.matrixAsSquares(lay.W[1:,i].reshape(lay.windowSizes)/wmagmax,maxSize=0.5,color=True)
        # plt.imshow(lay.W[1:,i].reshape(lay.windowSizes), interpolation='nearest',cmap=plt.cm.bwr,clim=(-wmagmax,wmagmax))
        # plt.axis('off')
    # plt.subplot(1,2,2)
    # lay = convClassifier.layers[1]
    # for i in range(lay.nUnits):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(lay.W[1:,i].reshape(lay.windowSizes), interpolation='nearest',cmap=plt.cm.binary)
    #     plt.axis('off')
    
        

    plt.figure(7)
    plt.clf()
    plt.plot(convClassifier.getErrorTrace())
    
    # print("Changing np.set_printoptions")
    # np.set_printoptions(linewidth=120,precision=2,threshold=np.nan)

