from mlpython.learners.generic import Learner
import numpy as np
import theano as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class NeuralNetwork(Learner):
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
                 sizes=[200,100,50],
                 L2=0,
                 L1=0,
                 seed=1234,
                 tanh=False,
                 parameter_initialization=None,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1
        self.seed=seed
        self.tanh=tanh
        self.parameter_initialization = parameter_initialization
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0
        self.params = []

    def initialize(self, input_size, n_classes, batchsize):
        """
        This method allocates memory for the fprop/bprop computations (DONE)
        and initializes the parameters of the neural network (TODO)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        n_hidden_layers = len(self.sizes)

        #########################
        # Initialize parameters #
        #########################

        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate

        self.inputTensor = T.tensor.matrix("input").reshape((batchsize,1,72,72))
        targets = T.tensor.ivector('target')


        filter_shapes = [(self.sizes[0], 1, 3, 3), (self.sizes[1], self.sizes[0], 4, 4),
                         (self.sizes[2], self.sizes[1], 5, 5), (self.sizes[3], self.sizes[2], 6, 6)]

        C1 = abs(self.createConvolutionLayer(self.inputTensor, filter_shapes[0], (batchsize,1,72,72)))
        S2 = self.createPoolingLayer(C1, (2, 2), filter_shapes[0])
        C3 = abs(self.createConvolutionLayer(S2, filter_shapes[1], (batchsize, self.sizes[0], 35, 35)))
        S4 = self.createPoolingLayer(C3, (2, 2), filter_shapes[1])
        C5 = abs(self.createConvolutionLayer(S4, filter_shapes[2], (batchsize, self.sizes[1], 16, 16)))
        S6 = self.createPoolingLayer(C5, (2, 2), filter_shapes[2])
        C7 = abs(self.createConvolutionLayer(S6, filter_shapes[3], (batchsize, self.sizes[2], 6, 6)))


        output_layer = self.createSigmoidLayer(C7.flatten(2), self.sizes[-1], 1)
        cost = self.training_loss(output_layer, targets)
        nll = -T.tensor.log(output_layer)#-T.tensor.mean(T.tensor.log(output_layer))
        grads = T.tensor.grad(cost, self.params)

        n_updates = T.shared(0.)

        #updates = [(param_i, param_i - self.lr * grad_i) for param_i, grad_i in zip(self.params, grads)]
        updates = [self.update_param(param_i, grad_i, n_updates) for param_i, grad_i in zip(self.params, grads)]
        updates += [(n_updates, n_updates + 1.)]

        self.train_batch = T.function([self.inputTensor, targets], None, updates=updates,
                                      allow_input_downcast=True)

        self.cost_function = T.function([self.inputTensor], nll, allow_input_downcast=True)

        self.pred_y = T.function([self.inputTensor], T.tensor.argmax(output_layer, axis=1),
                                 allow_input_downcast=True)
        self.theano_fprop = T.function([self.inputTensor], output_layer,allow_input_downcast=True)

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
        print(filter_shape)

        #b_values = np.zeros((filter_shape[0],), dtype=T.config.floatX)
        #b = T.shared(b_values)
        #conv_out = conv_out + b.dimshuffle('x', 0, 'x', 'x')

        self.params.append(W)
        #self.params.append(b)
        return conv_out

    def createPoolingLayer(self, input, poolsize, prev_filter_shape):
        pool_out = downsample.max_pool_2d(input=input, ds=poolsize, ignore_border=True)

        b_values = np.zeros((prev_filter_shape[0],), dtype=T.config.floatX)
        b = T.shared(value=b_values, borrow=True)
        self.params.append(b)

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

    #TODO THEANO-IZE
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
                score = self.train_batch(input.reshape(batchsize,1,72,72), target)
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


    #TODO THEANO-IZE
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


