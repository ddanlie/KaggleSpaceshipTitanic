from copy import deepcopy
import numpy as np
import math



def mean_squared_error(yid, ypr):
    return np.mean(np.power(yid-ypr,2))

def root_mean_squared_error(yid, ypr):
    return np.sqrt(np.mean(np.power(yid-ypr,2))) 

def mean_absolute_error(yid, ypr):
    return np.mean(np.abs(yid-ypr))

def accuracy_metric(yid, ypr):
    return np.bincount(yid==ypr)[1] / yid.shape[0]
    
def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


class Operation:

    def __init__(self):
        pass

    def forward(self, input: np.ndarray, inference: bool = False):
        self._input = input
        self.output = self._output(inference)
        return self.output

    def backward(self, gradient_input: np.ndarray):
        assert self.output.shape == gradient_input.shape
        self.gradient_output = self._gradient_output(gradient_input)
        assert self.gradient_output.shape == self._input.shape
        return self.gradient_output

    def _output(self, inference):
        raise NotImplementedError()

    def _gradient_output(self, gradient_input: np.ndarray):
        raise NotImplementedError()

    
class ParamOperation(Operation):

    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param

    def backward(self, gradient_input: np.ndarray):
        assert self.output.shape == gradient_input.shape
        self.gradient_output = self._gradient_output(gradient_input)
        self.gradient_param_output = self._gradient_param_output(gradient_input)
        assert self.gradient_output.shape == self._input.shape
        assert self.gradient_param_output.shape == self.param.shape
        return self.gradient_output

    def _gradient_param_output(self, gradient_input: np.ndarray):
        raise NotImplementedError()


class Sigmoid(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference):
        return sigmoid_func(self._input)

    def _gradient_output(self, gradient_input: np.ndarray):
        return gradient_input*(self.output*(1- self.output))

class Tanh(Operation):

    def __init__(self):
        super().__init__()
    
    def _output(self, inference):
        return tanh(self._input)
    
    def _gradient_output(self, gradient_input: np.ndarray):
        return gradient_input*(1 - np.power(tanh(self._input), 2))


class Linear(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference):
        return self._input

    def _gradient_output(self, gradient_input: np.ndarray):
        return gradient_input


class WeightMultiply(ParamOperation):

    def __init__(self, weights: np.ndarray):
        super().__init__(weights)

    def _output(self, inference):
        return np.dot(self._input, self.param)

    def _gradient_output(self, gradient_input: np.ndarray):
        return np.dot(gradient_input, np.transpose(self.param))

    def _gradient_param_output(self, gradient_input: np.ndarray):
        return np.dot(np.transpose(self._input), gradient_input)


class BiasAdd(ParamOperation):

    def __init__(self, b: np.ndarray, biasPerLayer: bool = False):
        self._bpl = biasPerLayer
        if biasPerLayer:
            assert b.ndim == b.shape[0] == 1
        super().__init__(b)

    def _output(self, inference):
        return self._input + self.param    

    def _gradient_output(self, gradient_input: np.ndarray):
        return np.ones(self._input.shape) * gradient_input

    def _gradient_param_output(self, gradient_input: np.ndarray):
        if self._bpl:
            return np.array([np.sum(np.sum(gradient_input, axis=0), axis=0)])
        else:
            return np.sum(np.ones(self.param.shape) * gradient_input, axis=0)


class Dropout(Operation):

    def __init__(self, p: float):
        super().__init__()
        self._p = p
    
    def _output(self, inference: bool):
        if inference:
            return self._input * (1-self._p)
        else:
            self._mask = np.random.binomial(1, self._p, self._input.shape)
            return self._input * self._mask
        
    def _gradient_output(self, gradient_input: np.ndarray):
        return gradient_input * self._mask

class Layer:

    def __init__(self, neurons_count: int, activation: Operation, dropout: float=1.0):
        self._activation = activation
        self._ncnt = neurons_count
        self.params: list[np.ndarray] = []
        self.param_gradients: list[np.ndarray] = []
        self._ops: list[Operation] = []
        self.setup = True

    def _setup_layer(self, input: np.ndarray):
        raise NotImplementedError()

    def forward(self, input: np.ndarray, inference: bool = False):
        if self.setup:
            self._setup_layer(input)
            self.setup = False
        self._input = input
        self._output = input
        for operation in self._ops:
            self._output = operation.forward(self._output, inference)
        self.__extract_params()
        return self._output

    def backward(self, input_gradient: np.ndarray):
        assert self._output.shape == input_gradient.shape
        output_gradient = input_gradient
        for operation in reversed(self._ops):
            output_gradient = operation.backward(output_gradient)
        self.__extract_param_gradients()
        return output_gradient
    
    def __extract_param_gradients(self):
        self.param_gradients = []
        for op in self._ops:
            if issubclass(op.__class__, ParamOperation):
                self.param_gradients.append(op.gradient_param_output)

    def __extract_params(self):
        self.params = []
        for op in self._ops:
            if issubclass(op.__class__, ParamOperation):
                self.params.append(op.param)


class Dense(Layer):

    def __init__(self, neurons_count: int, activation: Operation, dropout: float=1.0,
        weight_init = "", biasPerLayer=False, seed: int=1):
        super().__init__(neurons_count, activation, dropout)
        self._bpl = biasPerLayer
        self._seed = seed
        self._weight_init = weight_init
        self._dropout = dropout

    def _setup_layer(self, input: np.ndarray):
        np.random.seed(self._seed)
        variance = 1.0
        if self._weight_init == "glorot":
            variance = 2/(input.shape[0] + self._ncnt)
        self.params = []
        self.params.append(variance * np.random.randn(input.shape[1], self._ncnt))
        if self._bpl:
            self.params.append(variance * np.random.randn())
        else:
            self.params.append(variance * np.random.randn(self._ncnt))
        
        if  self._dropout < 1.0:
            self._ops = [WeightMultiply(self.params[0]), Dropout(self._dropout), 
                         BiasAdd(self.params[1], self._bpl), self._activation]
        else:
            self._ops = [WeightMultiply(self.params[0]), BiasAdd(self.params[1], self._bpl), self._activation]


class Loss:

    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray):
        assert prediction.shape == target.shape
        self._prediction = prediction
        self._target = target
        loss = self._output()
        return loss

    def backward(self):
        self.output_gradient = self._output_gradient()
        assert self._prediction.shape == self.output_gradient.shape
        return self.output_gradient

    def _output(self):
        raise NotImplementedError()

    def _output_gradient(self):
        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self):
        super().__init__()

    def _output(self):
        return np.sum(np.power(self._target - self._prediction, 2))/self._prediction.shape[0]

    def _output_gradient(self):
        return 2.0 * (self._target - self._prediction) / self._prediction.shape[0]


class SoftmaxCrossEntropyError(Loss):
    
    def __init__(self):
        super().__init__()

    def _output(self):
        self.softmax_prediction = softmax(self._prediction)
        self.softmax_prediction = np.clip(self.softmax_prediction, 1e-5, 1-(1e-5))
        cross_entropy_loss = -1.0 * self._target * np.log(self.softmax_prediction) - \
                            (1.0-self._target) * np.log(1.0 - self.softmax_prediction)
        return np.sum(cross_entropy_loss)

    def _output_gradient(self):
        return self.softmax_prediction - self._target


class NeuralNet:

    def __init__(self, layers: list[Layer], loss: Loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x_batch: np.ndarray, inference: bool = False):
        y = x_batch
        for layer in self.layers:
            y = layer.forward(y, inference)
        return y

    def backward(self, loss_gradient: np.ndarray):
        gradient_output = loss_gradient
        for layer in reversed(self.layers):
            gradient_output = layer.backward(gradient_output)
        
    def train_batch(self, x_batch, y_batch):
        prediction = self.forward(x_batch)
        loss = self.loss.forward(prediction, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_gradients(self):
        for layer in self.layers:
            yield from layer.param_gradients
    

class Optimizer:

    def __init__(self, lr: float):
        self._lr = lr
        self._first = True

    def _step(self):
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, lr: float):
        super().__init__(lr)
        

    def _step(self):
        for (param, param_gradient) in zip(self._net.params(), self._net.param_gradients()):
            param -= self._lr * param_gradient


class SGDMomentum(Optimizer):

    def __init__(self, lr: float, lrend: float, momentum: float, dec_type: str = ""):
        super().__init__(lr)
        self._momentum = momentum
        self._lrend = lrend 
        self._dec_type = dec_type    


    def _step(self):
        
        if self._first:
            self._first = False
            self.velocities = [np.zeros_like(param) for param in self._net.params()]

        for velocity, param, param_gradient in \
            zip(self._net.params(), self._net.param_gradients(), self.velocities):

            velocity *= self._momentum
            velocity += param_gradient
            param -= self._lr * param_gradient

    def _dec_lr(self, epochs: int, epnum: int):

        if self._dec_type == "lin":
            self._lr -= (self._lr - self._lrend)*(epnum/epochs)
        elif self._dec_type == "exp":
            self._lr = self._lr*math.pow(math.pow(self._lrend/self._lr, 1/(epochs-1)), epnum)



        


class Trainer:

    def __init__(self, net: NeuralNet, optimizer: Optimizer):
        self._net = net
        self._optimizer = optimizer
        self._best_loss = 1e9
        setattr(self._optimizer, '_net', self._net)
    
    def generate_batches(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        assert x.shape[0] == y.shape[0]
        for bnum in range(int(x.shape[0]/batch_size)):
            x_b, y_b = x[bnum*batch_size:(bnum+1)*batch_size], y[bnum*batch_size:(bnum+1)*batch_size]
            yield (x_b, y_b)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray,
                    epochs: int, batch_size: int, eval_every: int, restart: bool=True):

        if restart:
            for layer in self._net.layers:
                layer.setup = True

        for epnum in range(epochs):

            x_train, y_train = np.random.permutation(x_train), np.random.permutation(y_train)
            batch_generator = self.generate_batches(x_train, y_train, batch_size)
            for x_b, y_b in batch_generator:
                self._net.train_batch(x_b, y_b)
                self._optimizer._step()
            
            if isinstance(self._optimizer.__class__, SGDMomentum):
                self._optimizer._dec_lr(epochs, epnum)

            if (epnum+1) % eval_every == 0:
                last_model = deepcopy(self._net)
                
                test_predictions = last_model.forward(x_test, inference=True)
                test_loss = self._net.loss.forward(test_predictions, y_test)
                if test_loss < self._best_loss:
                    self._best_loss = test_loss
                else:
                    self.net = last_model
                    setattr(self._optimizer, '_net', self.net)
                    break


class LinearRegression():

    def __init__(self, weights: np.ndarray, a, b, train_x: np.ndarray=None, train_y: np.ndarray=None):
        assert weights.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        self.train_x = train_x
        self.train_y = train_y
        self.weights = weights
        self.a = a
        self.b = b

    def __gradient_descent(self, dfdw, dfdb):
        dfdb = dfdb.sum()
        self.weights -= self.a*np.reshape(dfdw, self.weights.shape)
        self.b -= self.a*dfdb

    def predict(self, x):
        p = np.dot(x, self.weights)
        pb = p+self.b
        return pb
    
    def train(self, epochs, batch_size=1, e=0.1):
        for ep in range(epochs):
            err = 0
            for bnum in range(int(self.train_x.shape[0]/batch_size)):
                x_train_batch = self.train_x[bnum*batch_size : batch_size*(bnum+1)]
                y_train_batch = self.train_y[bnum*batch_size : batch_size*(bnum+1)] 
                ypr = self.predict(x_train_batch)
                dfdm = np.array([(-2*(yi-pbi))/ypr.shape[0] for pbi, yi in zip(ypr, y_train_batch) ])
                dfdw = np.dot(np.transpose(x_train_batch), dfdm)
                dfdb = dfdm
                self.__gradient_descent(dfdw, dfdb)
                err += mean_squared_error(y_train_batch, ypr)
            if(err/batch_size < e):
                return err/batch_size


class LogisticRegression:

    def __init__(self, weights, a, b, train_x=None, train_y=None):
        self.train_x = np.array([np.append(train_x[i], 1) for i in range(train_x.shape[0])])
        self.train_y = train_y
        self.weights = np.append(weights, b)
        self.a = a
        self.b = b

    def __gradient_descent(self, yid, ypr, x):
        for i in range(len(self.weights)):
            derw = 0
            for k in range(len(yid)):
                derw += -x[k][i]*(yid[k]*(1-ypr[k]) - (1-yid[k])*ypr[k])
                    
            self.weights[i] -= derw*self.a

    def predict(self, x):
        if(x.shape[0] != self.weights.shape[0]):
            x = np.append(x, 1)
        return sigmoid_func(np.dot(self.weights, x))
    
    def train(self, epochs, batch_size, e):
        for i in range(epochs):
                accuracy = 0
                ypr = np.zeros(batch_size)
                idx = 0
                for j in range(self.train_y.shape[0]):
                    ypr[idx] = self.predict(self.train_x[j])
                    idx += 1
                    if (j+1) % batch_size == 0:
                        idx = batch_size*((j)//batch_size)
                        idx2 = self.train_y.shape[0]
                        if idx + batch_size < idx2:
                            idx2 = idx + batch_size
                        self.__gradient_descent(self.train_y[idx:idx2], ypr, self.train_x[idx:idx2])
                        accuracy += accuracy_metric(self.train_y[idx:idx2], [round(i) for i in ypr])
                        idx = 0
                accuracy /= ((self.train_y.shape[0]+1) / batch_size)
                if accuracy > 1-e:
                    return 1-accuracy
        return 1-accuracy