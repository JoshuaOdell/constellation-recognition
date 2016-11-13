import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch  # for Arc

######################################################################
# Machine Learning Utilities. 
#
#  makeStandardize
#  makeIndicatorVars
#  segmentize
#  confusionMatrix
#  printConfusionMatrix
#  partition
#  partitionsKFolds
#  rollingWindows
#  drawCascade (a neural network)
#  draw (a neural network)
#  matrixAsSquares
######################################################################


######################################################################

def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)


def makeIndicatorVars(T):
    """ Assumes argument is N x 1, N samples each being integer class label """
    return (T == np.unique(T)).astype(int)

######################################################################
# Build matrix of labeled time windows, each including nlags consecutive
# samples, with channels concatenated.  Keep only segments whose nLags
# targets are the same. Assumes target is in last column of data.
def segmentize(data,nLags):
    nSamples,nChannels = data.shape
    nSegments = nSamples-nLags+1
    # print 'segmentize, nSamples',nSamples,'nChannels',nChannels,'nSegments',nSegments
    # print 'data.shape',data.shape,'nLags',nLags
    segments = np.zeros((nSegments, (nChannels-1) * nLags + 1))
    k = 0
    for i in range(nSegments):
        targets = data[i:i+nLags,-1]
        # In previous versions had done this...
        #    to throw away beginning and end of each trial, and between trials
        #       if targets[0] != 0 and targets[0] != 32: 
        #            # keep this row
        if (np.all(targets[0] == targets)):
            allButTarget = data[i:i+nLags,:-1]
            # .T to keep all from one channel together
            segments[k,:-1] = allButTarget.flat
            segments[k,-1] = targets[0]
            k += 1
    return segments[:k,:]

def confusionMatrixOld(actual,predicted,classes):
    nc = len(classes)
    confmat = np.zeros((nc,nc))
    for ri in range(nc):
        trues = actual==classes[ri]
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predicted[trues] == classes[ci]) / float(np.sum(trues))
    return confmat


######################################################################

def confusionMatrix(actual,predicted,classes,probabilities=None,probabilityThreshold=None):
    nc = len(classes)
    if probabilities is not None:
        predictedClassIndices = np.zeros(predicted.shape,dtype=np.int)
        for i,cl in enumerate(classes):
            predictedClassIndices[predicted == cl] = i
        probabilities = probabilities[np.arange(probabilities.shape[0]), predictedClassIndices.squeeze()]
    confmat = np.zeros((nc,nc+2)) # for samples above threshold this class and samples this class
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        if probabilities is None:
            keep = trues
            predictedThisClassAboveThreshold = predictedThisClass
        else:
            keep = probabilities[trues] >= probabilityThreshold
            predictedThisClassAboveThreshold = predictedThisClass[keep]
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
        confmat[ri,nc] = np.sum(keep)
        confmat[ri,nc+1] = np.sum(trues)
    return confmat

def printConfusionMatrix(confmat,classes):
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('%s' % '------'*len(classes))
    for i,t in enumerate(classes):
        print('%2d |' % (t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('%5.1f' % (100*confmat[i,i1]), end='')
        print('   (%d / %d)' % (int(confmat[i,len(classes)]), int(confmat[i,len(classes)+1])))

######################################################################

def partition(X,T,fractions,classification=False):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),classification=True)
      X is nSamples x nFeatures.
      fractions can have just two values, for partitioning into train and test only
      If classification=True, T is target class as integer. Data partitioned
        according to class proportions.
        """
    trainFraction = fractions[0]
    if len(fractions) == 2:
        # Skip the validation step
        validateFraction = 0
        testFraction = fractions[1]
    else:
        validateFraction = fractions[1]
        testFraction = fractions[2]
        
    rowIndices = np.arange(X.shape[0])
    np.random.shuffle(rowIndices)
    
    if classification != 1:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        Xtrain = X[rowIndices[:nTrain],:]
        Ttrain = T[rowIndices[:nTrain],:]
        if nValidate > 0:
            Xvalidate = X[rowIndices[nTrain:nTrain+nValidate],:]
            Tvalidate = T[rowIndices[nTrain:nTrain:nValidate],:]
        Xtest = X[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        Ttest = T[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        
    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        trainIndices = []
        validateIndices = []
        testIndices = []
        for c in classes:
            # row indices for class c
            cRows = np.where(T[rowIndices,:] == c)[0]
            # collect row indices for class c for each partition
            n = len(cRows)
            nTrain = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate
            trainIndices += rowIndices[cRows[:nTrain]].tolist()
            if nValidate > 0:
                validateIndices += rowIndices[cRows[nTrain:nTrain+nValidate]].tolist()
            testIndices += rowIndices[cRows[nTrain+nValidate:nTrain+nValidate+nTest]].tolist()
        Xtrain = X[trainIndices,:]
        Ttrain = T[trainIndices,:]
        if nValidate > 0:
            Xvalidate = X[validateIndices,:]
            Tvalidate = T[validateIndices,:]
        Xtest = X[testIndices,:]
        Ttest = T[testIndices,:]
    if nValidate > 0:
        return Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
    else:
        return Xtrain,Ttrain,Xtest,Ttest

######################################################################

def partitionsKFolds(X,T,K,validation=True,shuffle=True,classification=True):
    '''Returns Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
      or
       Xtrain,Ttrain,Xtest,Ttest if validation is False
    Build dictionary keyed by class label. Each entry contains rowIndices and start and stop
    indices into rowIndices for each of K folds'''
    global folds
    if not classification:
        print('Not implemented yet.')
        return
    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)
    folds = {}
    classes = np.unique(T)
    for c in classes:
        classIndices = rowIndices[np.where(T[rowIndices,:] == c)[0]]
        nInClass = len(classIndices)
        nEach = int(nInClass / K)
        starts = np.arange(0,nEach*K,nEach)
        stops = starts + nEach
        stops[-1] = nInClass
        # startsStops = np.vstack((rowIndices[starts],rowIndices[stops])).T
        folds[c] = [classIndices, starts, stops]

    for testFold in range(K):
        if validation:
            for validateFold in range(K):
                if testFold == validateFold:
                    continue
                trainFolds = np.setdiff1d(range(K), [testFold,validateFold])
                rows = rowsInFold(folds,testFold)
                Xtest = X[rows,:]
                Ttest = T[rows,:]
                rows = rowsInFold(folds,validateFold)
                Xvalidate = X[rows,:]
                Tvalidate = T[rows,:]
                rows = rowsInFolds(folds,trainFolds)
                Xtrain = X[rows,:]
                Ttrain = T[rows,:]
                yield Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
        else:
            # No validation set
            trainFolds = np.setdiff1d(range(K), [testFold])
            rows = rowsInFold(folds,testFold)
            Xtest = X[rows,:]
            Ttest = T[rows,:]
            rows = rowsInFolds(folds,trainFolds)
            Xtrain = X[rows,:]
            Ttrain = T[rows,:]
            yield Xtrain,Ttrain,Xtest,Ttest
            

def rowsInFold(folds,k):
    allRows = []
    for c,rows in folds.items():
        classRows, starts, stops = rows
        allRows += classRows[starts[k]:stops[k]].tolist()
    return allRows

def rowsInFolds(folds,ks):
    allRows = []
    for k in ks:
        allRows += rowsInFold(folds,k)
    return allRows

######################################################################

def trainValidateTestKFolds(trainf,evaluatef,X,T,parameterSets,nFolds,
                            shuffle=False,classification=False,verbose=False):
    # Randomly arrange row indices
    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)
    if classification:
        isNewBetterThanOld = lambda new,old: new > old
        folds = {}
        classes = np.unique(T)
        for c in classes:
            classIndices = rowIndices[np.where(T[rowIndices,:] == c)[0]]
            nInClass = len(classIndices)
            nEach = int(nInClass / nFolds)
            starts = np.arange(0,nEach*nFolds,nEach)
            stops = starts + nEach
            stops[-1] = nInClass
            # startsStops = np.vstack((rowIndices[starts],rowIndices[stops])).T
            folds[c] = [classIndices, starts, stops]
    else:
        isNewBetterThanOld = lambda new,old: new < old
        # Calculate number of samples in each of the nFolds folds
        nSamples = X.shape[0]
        nEach = int(nSamples / nFolds)
        if nEach == 0:
            raise ValueError("trainValidateTestKFolds: Number of samples in each fold is 0.")
        # Calculate the starting and stopping row index for each fold.
        # Store in startsStops as list of (start,stop) pairs
        starts = np.arange(0,nEach*nFolds,nEach)
        stops = starts + nEach
        stops[-1] = nSamples
        startsStops = list(zip(starts,stops))
        
    # Repeat with testFold taking each single fold, one at a time
    results = []
    for testFold in range(nFolds):
        # Leaving the testFold out, for each validate fold, train on remaining
        # folds and evaluate on validate fold. Collect theseRepeat with validate
        # Construct Xtest and Ttest
        if classification:
            rows = rowsInFold(folds,testFold)
        else:
            a,b = startsStops[testFold]
            rows = rowIndices[a:b]
        Xtest = X[rows,:]
        Ttest = T[rows,:]

        bestParms = None
        for parms in parameterSets:
            validateEvaluationSum = 0
            for validateFold in range(nFolds):
                if testFold == validateFold:
                    continue
                # Construct Xtrain and Ttrain
                trainFolds = np.setdiff1d(range(nFolds), [testFold,validateFold])
                if classification:
                    rows = rowsInFolds(folds,trainFolds)
                else:
                    rows = []
                    for tf in trainFolds:
                        a,b = startsStops[tf]                
                        rows += rowIndices[a:b].tolist()
                Xtrain = X[rows,:]
                Ttrain = T[rows,:]
                # Construct Xvalidate and Tvalidate
                if classification:
                    rows = rowsInFold(folds,validateFold)
                else:
                    a,b = startsStops[validateFold]
                    rows = rowIndices[a:b]
                Xvalidate = X[rows,:]
                Tvalidate = T[rows,:]

                model = trainf(Xtrain,Ttrain,parms)
                validateEvaluation = evaluatef(model,Xvalidate,Tvalidate)
                validateEvaluationSum += validateEvaluation
            validateEvaluation = validateEvaluationSum / (nFolds-1)
            if bestParms is None or isNewBetterThanOld(validateEvaluation,bestValidationEvaluation): 
                bestParms = parms
                bestValidationEvaluation = validateEvaluation


        newXtrain = np.vstack((Xtrain,Xvalidate))
        newTtrain = np.vstack((Ttrain,Tvalidate))
        model = trainf(newXtrain,newTtrain,bestParms)
        trainEvaluation = evaluatef(model,newXtrain,newTtrain)
        testEvaluation = evaluatef(model,Xtest,Ttest)

        resultThisTestFold = [bestParms, trainEvaluation,
                              bestValidationEvaluation, testEvaluation]
        results.append(resultThisTestFold)
        if verbose:
            print(resultThisTestFold)
    return results


######################################################################

def rollingWindows(a, window, noverlap=0):
    '''a: multi-dimensional array, last dimension will be segmented into windows
    window: number of samples per window
    noverlap: number of samples of overlap between consecutive windows

    Returns:
    array of windows'''
    shapeUpToLastDim = a.shape[:-1]
    nSamples = a.shape[-1]
    if noverlap > window:
        print('rollingWindows: noverlap > window, so setting to window-1')
        noverlap = window-1 # shift by one
    nShift = window - noverlap
    nWindows = int((nSamples - window + 1) / nShift)
    newShape = a.shape[:-1] + (nWindows, window)
    strides = a.strides[:-1] + (a.strides[-1]*nShift, a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=newShape, strides=strides)

######################################################################
# Associated with  neuralnetworks.py
# Draw a neural network with weights in each layer as a matrix
######################################################################

def drawCascade(W, inputNames = None, outputNames = None, gray = False):
    nLayers = len(W)


    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = 0
    if inputNames:
        xlim = max(map(len,inputNames))/4.0 if inputNames else 1
    ylim = 0
    
    xlim += sum([w.shape[1] for w in W]) + 0.5 * len(W)
    ylim += W[-1].shape[0] 

    # Add space for output names
    if outputNames:
        xlim += round(max(map(len,outputNames))/4.0)
    ylim += 0.25

    maxw = max([np.max(np.abs(w)) for w in W])
    largestDimNWeights = max( sum([W[0].shape[0]] + [w.shape[1] for w in W[:-1]]),
                              sum([w.shape[1] for w in W]))
    largestDim = max(xlim,ylim)
    scaleW = 200 /  largestDimNWeights / maxw
    print("C",largestDim,largestDimNWeights,maxw,scaleW)
    
    ax = plt.gca()
    x0 = 1
    y0 = 0 # to allow for constant input to first layer
    # First Layer
    if inputNames:
        # addx = max(map(len,inputNames))*0.1
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0-len(n)*0.2, y, n)
            x0 = max([1,max(map(len,inputNames))/4.0])

    for li in range(nLayers):
        Wi = W[li]
        ni,no = Wi.shape
        # Vertical layer. Origin is upper left.
        # Constant input
        if li == 0:
            ax.text(x0-0.2, y0+0.5, '1')
        if li == 0:
            for i in range(ni):
                # ax.plot((x0,x0+no-0.5), (y0+i+0.5, y0+i+0.5),color='gray')
                ax.plot((x0,xlim), (y0+i+0.5, y0+i+0.5),color='gray')
        # output lines
        for i in range(no):
            ax.plot((x0+1+i-0.5, x0+1+i-0.5), (y0, y0+ni+i),color='gray')
            ax.plot(x0+1+i-0.5, y0+ni,'v',markersize=12,color='gray')
            # corners
            if li < nLayers-1:
                ax.add_patch(pltpatch.Arc((x0+1+i-0.5+0.5, y0+ni+i), 1,1, angle=0, theta1=90, theta2=180, color='gray'))
        # cell "bodies"
        xs = x0 + np.arange(no) + 0.5
        ys = np.array([y0+ni+0.5]*no)
        # ax.scatter(xs,ys,marker='d',s=100,c='gray')
        # outputs turn right and go to all next layers
        ys += range(no)
        if li < nLayers-1:
            for i in range(no):
                # pdb.set_trace()
                ax.plot(np.vstack((xs+0.5,[xlim]*no)), np.tile(ys,(2,1)), color='gray')
        # weights
        if gray:
            colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
        else:
            colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
        xs = np.arange(no)+ x0+0.5
        ys = np.arange(ni)+ y0 + 0.5
        aWi = (np.abs(Wi) * scaleW) **2
        coords = np.meshgrid(xs,ys)
        #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
        ax.scatter(coords[0],coords[1],marker='s',s=aWi,c=colors)
        # leave y0 at 0 for all layers
        # y0 += ni + 1
        x0 += no ## shift for next layer's constant input

    # Last layer output labels 
    if outputNames:
        y = y0+0.6
        for n in outputNames:
            y += 1
            ax.text(x0+0.2, y, n)
    ax.axis([0,xlim, ylim,0])
    ax.axis('off')
    # ax.figure.canvas.draw()

def draw(W, inputNames = None, outputNames = None, gray = False):

    def isOdd(x):
        return x % 2 != 0

    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len,inputNames))/4.0 if inputNames else 1
    ylim = 0
    
    for li in range(nLayers):
        ni,no = W[li].shape  #no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni,no = W[nLayers-1].shape  #no means number outputs this layer
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len,outputNames))/4.0)


    maxw = max([np.max(np.abs(w)) for w in W])
    largestDimNWeights = max( sum([w.shape[i%2] for i,w in enumerate(W)]),
                              sum([w.shape[1-i%2] for i,w in enumerate(W)]))

    largestDim = max(xlim,ylim)
    scaleW = 2000 /  largestDimNWeights / maxw
    # print(largestDim,largestDimNWeights,maxw,scaleW)

    ax = plt.gca()

    x0 = 1
    y0 = 0 # to allow for constant input to first layer
    # First Layer
    if inputNames:
        # addx = max(map(len,inputNames))*0.1
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0-len(n)*0.2, y, n)
            x0 = max([1,max(map(len,inputNames))/4.0])

    for li in range(nLayers):
        Wi = W[li]
        ni,no = Wi.shape
        if not isOdd(li):
            # Odd layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1')
            for i in range(ni):
                ax.plot((x0,x0+no-0.5), (y0+i+0.5, y0+i+0.5),color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0+1+i-0.5, x0+1+i-0.5), (y0, y0+ni+1),color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            ax.scatter(xs,ys,marker='v',s=1000,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(no)+ x0+0.5
            ys = np.arange(ni)+ y0 + 0.5
            aWi = (np.abs(Wi) * scaleW)**2
            coords = np.meshgrid(xs,ys)
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            y0 += ni + 1
            x0 += -1 ## shift for next layer's constant input
        else:
            # Even layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1')
            # input lines
            for i in range(ni):
                ax.plot((x0+i+0.5,  x0+i+0.5), (y0,y0+no-0.5),color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0, x0+ni+1), (y0+i+0.5, y0+i+0.5),color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5]*no)
            ys = y0 + 0.5 + np.arange(no)
            ax.scatter(xs,ys,marker='>',s=1000,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs,ys)
            aWi = (np.abs(Wi) * scaleW)**2
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            x0 += ni + 1
            y0 -= 1 ##shift to allow for next layer's constant input

    # Last layer output labels 
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n)
    ax.axis([0,xlim, ylim,0])
    ax.axis('off')

# import pdb

def matrixAsSquares(W, maxSize = 200, color = False):

    if color:
        facecolors = np.array(["red","green"])[(W.flat >= 0)+0]
        edgecolors = np.array(["red","green"])[(W.flat >= 0)+0]
    else:
        facecolors = np.array(["black","none"])[(W.flat >= 0)+0]
        edgecolors = np.array(["black","black"])[(W.flat >= 0)+0]

    xs = np.arange(W.shape[1]) #+ x0+0.5
    ys = np.arange(W.shape[0]) #+ y0 + 0.5
    maxw = np.max(np.abs(W))
    aWi = (np.abs(W)/maxw) * maxSize
    coords = np.meshgrid(xs,ys)
    #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
    ax = plt.gca()
    # pdb.set_trace()
    for x, y, width, fc,ec  in zip(coords[0].flat,coords[1].flat, aWi.flat, facecolors, edgecolors):
        ax.add_patch(pltpatch.Rectangle((x+maxSize/2-width/2, y+maxSize/2)-width/2, width,width,facecolor=fc,edgecolor=ec))
    # plt.scatter(coords[0],coords[1],marker='s',s=aWi,
    #            facecolor=facecolors,edgecolor=edgecolors)
    plt.xlim(-.5,W.shape[1]+0.5)
    plt.ylim(-.5,W.shape[0]-0.5)
    plt.axis('tight')

if __name__ == '__main__':
    plt.ion()

    plt.figure(1)
    plt.clf()

    plt.subplot(2,1,1)
    w = np.arange(-100,100).reshape(20,10)
    matrixAsSquares(w,200,color=True)

    plt.subplot(2,1,2)
    w = np.arange(-100,100).reshape(20,10)
    matrixAsSquares(w,100,color=False)

