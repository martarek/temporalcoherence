from mlpython.learners.generic import Learner
import numpy as np
import theano as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class TemporalNeuralNetwork(Learner):
    """
    Neural network for classification.

    Option ``lr`` is the learning rate.

    Option ``dc`` is the decrease constante for the learning rate.

    Option ``sizes`` is the list of hidden layer sizes.

    Option ``L2`` is the L2 regularization weight (weight decay).

    Option ``L1`` is the L1 regularization weight (weight decay).

    Option ``seed`` is the seed of the random number generator.

    Option ``tanh`` is a boolean indicating whether to use the
    hyperbolic tangent activation function (True) instead of the
    sigmoid activation function (True).

    Option ``parameter_initialization`` is a pair of lists,
    giving the initializations for the biases (first list)
    and the weight matrices (second list). If ``None``,
    then a random initialization is used.

    Option ``n_epochs`` number of training epochs.

    **Required metadata:**

    * ``'input_size'``: Size of the input.
    * ``'targets'``: Set of possible targets.

    """


    def __init__(self,
                 lr=0.001,
                 dc=0,
                 sizes=(50,100,150,200),
                 seed=1234,
                 parameter_initialization=None,
                 deltaDistance = 1,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.seed=seed
        self.parameter_initialization = parameter_initialization
        self.n_epochs=n_epochs
        self.deltaDistance = deltaDistance

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0
        self.params = []
        self.train_batch = [0,0,0]
        self.FIRST_PHASE = 0
        self.SECOND_PHASE = 1
        self.THIRD_PHASE = 2


    def initialize(self, input_size, n_classes, batchsize):
        """
        This method allocates memory for the fprop/bprop computations (DONE)
        and initializes the parameters of the neural network (TODO)
        """
        self.n_classes = n_classes
        self.input_size = input_size

        #########################
        # Initialize parameters #
        #########################
        self.inputTensor1 = T.tensor.matrix("input1").reshape((batchsize,1,72,72))
        self.inputTensor2 = T.tensor.matrix("input2").reshape((batchsize,1,72,72))

        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate


        targets = T.tensor.ivector('target')


        filter_shapes = [(self.sizes[0], 1, 3, 3), (self.sizes[1], self.sizes[0], 4, 4),
                         (self.sizes[2], self.sizes[1], 5, 5), (self.sizes[3], self.sizes[2], 6, 6)]

        formattedSizes = [(batchsize,1,72,72),
           (batchsize, self.sizes[0], 35, 35),
           (batchsize, self.sizes[1], 16, 16),
           (batchsize, self.sizes[2], 6, 6)]
        nnet1 = []

        for i in range(7):
            if i % 2 != 0:
                nnet1.append(self.createPoolingLayer(nnet1[-1], (2, 2), filter_shapes[i/2]))
            else:
                if i == 0:
                    nnet1.append(abs(self.createConvolutionLayer(self.inputTensor1, filter_shapes[i/2], formattedSizes[i/2])))
                else:
                    nnet1.append(abs(self.createConvolutionLayer(nnet1[-1], filter_shapes[i/2], formattedSizes[i/2])))

        nnet2 = []

        for i in range(7):
            if i % 2 != 0:
                nnet2.append(self.createPoolingLayerUsingParams(nnet2[-1], (2, 2), self.params[i]))
            else:
                if i == 0:
                    nnet2.append(abs(self.createConvolutionLayerUsingParams(self.inputTensor2, filter_shapes[i/2], formattedSizes[i/2], self.params[i])))
                else:
                    nnet2.append(abs(self.createConvolutionLayerUsingParams(nnet2[-1], filter_shapes[i/2], formattedSizes[i/2],self.params[i])))

        output_layer = [self.createSigmoidLayer(nnet1[-1].flatten(2), self.sizes[-1], 1),
                        self.createSigmoidLayerUsingParams(nnet2[-1].flatten(2), self.params[-2],self.params[-1])]

        cost_FirstPhase = self.training_loss(output_layer[0], targets)

        cost_SecondPhase = self.similarLossFunction([nnet1[-1],nnet2[-1]], batchsize)

        cost_ThirdPhase = self.dissimilarLossFunction([nnet1[-1],nnet2[-1]], batchsize)



        grads_FirstPhase = T.tensor.grad(cost_FirstPhase, self.params)

        #We stop before the last layer, being the output layer, hence the self.params[:-2] (W and B)
        grads_SecondPhase =  T.tensor.grad(cost_SecondPhase, self.params[:-2])

        grads_ThirdPhase = T.tensor.grad(cost_ThirdPhase, self.params[:-2])

        n_updates = T.shared(0.)

        updates_FirstPhase = [self.update_param(param_i, grad_i, n_updates) for param_i, grad_i in zip(self.params, grads_FirstPhase)]
        updates_FirstPhase += [(n_updates, n_updates + 1.)]

        updates_SecondPhase = [self.update_param(param_i, grad_i, n_updates) for param_i, grad_i in zip(self.params[:-2], grads_SecondPhase)]
        updates_SecondPhase += [(n_updates, n_updates + 1.)]

        updates_ThirdPhase = [self.update_param(param_i, grad_i, n_updates) for param_i, grad_i in zip(self.params[:-2], grads_ThirdPhase)]
        updates_ThirdPhase += [(n_updates, n_updates + 1.)]

        self.train_batch[self.FIRST_PHASE] = T.function([self.inputTensor1, targets], None, updates=updates_FirstPhase,
                                                        allow_input_downcast=True)
        self.train_batch[self.SECOND_PHASE] = T.function([self.inputTensor1, self.inputTensor2], None, updates=updates_SecondPhase,
                                                         allow_input_downcast=True)
        self.train_batch[self.THIRD_PHASE] = T.function([self.inputTensor1, self.inputTensor2], None, updates=updates_ThirdPhase,
                                                        allow_input_downcast=True)


        testLayer = output_layer[1]
        inputTensorToSelect = self.inputTensor2

        nll = -T.tensor.log(testLayer)#-T.tensor.mean(T.tensor.log(output_layer))
        self.cost_function = T.function([inputTensorToSelect], nll, allow_input_downcast=True)
        self.pred_y = T.function([inputTensorToSelect], T.tensor.argmax(testLayer, axis=1),
                                 allow_input_downcast=True)
        self.theano_fprop = T.function([inputTensorToSelect], testLayer,allow_input_downcast=True)


    def similarLossFunction(self, layersOfInterest,batchSize):
        return (layersOfInterest[0] - layersOfInterest[1]).norm(1) /batchSize

    def dissimilarLossFunction(self, layersOfInterest, batchSize):
        return T.tensor.max((0, self.deltaDistance - (layersOfInterest[0] - layersOfInterest[1]).norm(1)))/ batchSize

    def update_param(self, param_i, grad_i, n_updates):
        return param_i, param_i - grad_i * (self.lr / (1. + (n_updates * self.dc)))

    def createConvolutionLayer(self, input, filter_shape, image_shape):

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = T.shared(
            np.asarray(
                self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=T.config.floatX
            ),
            borrow=True
        )

        conv_out = conv.conv2d(input=input,
                               filters=W,
                               filter_shape=filter_shape,
                               image_shape=image_shape
        )


        self.params.append(W)
        return conv_out

    def createConvolutionLayerUsingParams(self, input, filter_shape, image_shape, W):
        return conv.conv2d(input=input,
                           filters=W,
                           filter_shape=filter_shape,
                           image_shape=image_shape
        )


    def createPoolingLayer(self, input, poolsize, prev_filter_shape):

        pool_out = downsample.max_pool_2d(input=input, ds=poolsize, ignore_border=True)

        b_values = np.zeros((prev_filter_shape[0],), dtype=T.config.floatX)
        b = T.shared(value=b_values, borrow=True)
        self.params.append(b)

        return T.tensor.tanh(pool_out + b.dimshuffle('x', 0, 'x', 'x'))

    def createPoolingLayerUsingParams(self, input, poolsize, b):
        pool_out = downsample.max_pool_2d(input=input, ds=poolsize, ignore_border=True)
        return T.tensor.tanh(pool_out + b.dimshuffle('x', 0, 'x', 'x'))

    def createSigmoidLayer(self, input, nkerns, img_size):
        W = T.shared(
            value=np.zeros(
                (nkerns*img_size, self.n_classes),
                dtype=T.config.floatX
            ),
            name='sigmoid W',
            borrow=True
        )
        b = T.shared(
            value = np.zeros((self.n_classes,), dtype=T.config.floatX),
            name = 'sigmoid b',
            borrow=True
        )
        self.params.append(W)
        self.params.append(b)
        return T.tensor.nnet.softmax(T.tensor.dot(input, W) + b)

    def createSigmoidLayerUsingParams(self, input, W, b):
        return T.tensor.nnet.softmax(T.tensor.dot(input, W) + b)

    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until
        ``self.epoch == self.n_epochs``.

        If ``self.epoch == 0``, first initialize the model.
        """
        batchsize = trainset.metadata['minibatch_size']
        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            print "initialize ..."
            self.initialize(input_size, n_classes, batchsize)
            print "done"

        for it in range(self.epoch,self.n_epochs):
            for input, target in trainset:
                consecutivesFrames = trainset.data.getConsecutivesFrames(batchsize)
                nonConsecutivesFrames = trainset.data.getNonConsecutivesFrames(batchsize)
                self.train_batch[self.FIRST_PHASE](input.reshape(batchsize,1,72,72), target)
                self.train_batch[self.SECOND_PHASE](consecutivesFrames[0].reshape(batchsize,1,72,72), consecutivesFrames[1].reshape(batchsize,1,72,72))
                self.train_batch[self.THIRD_PHASE](nonConsecutivesFrames[0].reshape(batchsize,1,72,72),nonConsecutivesFrames[1].reshape(batchsize,1,72,72))
                self.n_updates += 1
        self.epoch = self.n_epochs


    def training_loss(self,output,target):
        """
        Returns the negative log likelyhood (NLL) for a given minibatch
        :param output: Theano tensor representing the output function of the preceding layer
        :param target: Vector that gives for each example the correct label
        :return: nll : Theano tensor representing the NLL
        """
        return -T.tensor.mean(T.tensor.log(output)[T.tensor.arange(target.shape[0]), target])


    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a Numpy 2D array of size
          len(dataset) by (nb of classes + 1)
        - the ith row of the array contains the outputs for the ith example
        - the outputs for each example should contain
          the predicted class (first element) and the
          output probabilities for each class (following elements)
        """

        outputs = np.zeros((len(dataset)*dataset.metadata['minibatch_size'], self.n_classes+1))
        t=0
        for input,target in dataset:
            input = input.reshape(dataset.metadata['minibatch_size'],1,72,72)
            preds = self.pred_y(input)
            nlls = self.cost_function(input)
            for output, nll in zip(preds, nlls):
                outputs[t,0] = output
                outputs[t,1:] = nll
                t += 1
        #    outputs[t,0] = self.pred_y(input)
        #    outputs[t,1:] = self.cost_function(input)#self.theano_fprop(input)
        #    t += 1

        return outputs


    def forget(self):
        pass

    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of
        those outputs for ``dataset``:
        - the errors should be a Numpy 2D array of size
          len(dataset) by 2
        - the ith row of the array contains the errors for the ith example
        - the errors for each example should contain
          the 0/1 classification error (first element) and the
          regularized negative log-likelihood (second element)
         """

        outputs = self.use(dataset)
        errors = np.zeros((len(dataset)*dataset.metadata['minibatch_size'], 2))

        t=0
        for input,targets in dataset:
            for target in targets:
                output = outputs[t,:]
                errors[t,0] = output[0] != target
                errors[t,1] = output[int(target) + 1]#self.training_loss(output[1:],target)
                t+=1

        return outputs, errors


