import numpy as np

class ParametrizedCircuitOptimizer:
    '''Base class. Not for instantiation.

    The child classes are not the actual gradient descent optimization methods
    but rather map out the circuit logic defined in child classes of VqeCircuit
    of circuit_logic.py. I.e. for each child class there is a child class of
    ParametrizedCircuitOptimizer. Also the evolution is tracked, i.e. the history of the cost
    function and parameters.
    '''
    def init(self, circuit, max_iter):
        self.circuit = circuit
        self.max_iter = max_iter
        self.iter = 0

    def step(self):
        pass

    def reset(self):
        pass

    def pick(self, optimizer, ini_parameters):
        '''Instantiates one of the gradient descent methods implemented below.

        Args:
            caller (ParametrizedCircuitOptimizer): within it the specified gradient descent method
                will be instantiated.
            optimizer (dict): a dictionary containing the name of the optimizer to use
                along with its parameters.
        '''
        if optimizer['name'] == 'Adam':
            self.optimizer = Adam(ini_parameters, optimizer)
        elif optimizer['name'] == 'GradientDescent':
            self.optimizer = GradientDescent(ini_parameters, optimizer)
        elif optimizer['name'] == 'RateDecayOnPlateau':
            self.optimizer = RateDecayOnPlateau(ini_parameters, optimizer)
        else:
            raise ValueError('No optimizer {} known.'.format(optimizer))

class McCleanOpt(ParametrizedCircuitOptimizer):
    '''Optimizer for McClean circuits.

    Args:
        circuit (McClean): instantiation of a McClean circuit.
        optimizer (dict): a dictionary containing the name of the optimizer to use
            along with its parameters.
        max_iter (int): maximum number of steps (default is 1000).
    Keyword Args:
        ini_parameters (numpy.ndarray): initial parameters of the circuit. If none
            are given, the current angles of the circuit are used.
    '''
    def __init__(self, circuit, optimizer, max_iter=1000, **kwargs):
        ParametrizedCircuitOptimizer.init(self, circuit, max_iter)
        # set up history memory
        self.param_history = np.zeros([self.max_iter, circuit.lnum, circuit.qnum], dtype='double')
        self.cost_history = np.zeros(self.max_iter, dtype='double')
        # get kwargs
        ini_parameters = kwargs.get('ini_parameters', circuit.angles)
        self.param_history[0] = ini_parameters
        self.circuit.angles = ini_parameters
        # set up optimizer
        self.optimizer_info = optimizer
        self.pick(optimizer, ini_parameters)

    def __str__(self):
        return self.optimizer_info.__str__()

    def step(self, shot_num=0, dense_mode=True, component_sampling=False):
        if self.iter >= self.max_iter:
            print('Maximum amount of iterations reached: {}.'.format(self.max_iter))
        if shot_num == 0:
            if component_sampling == True:
                e, g = self.circuit.grad_run_with_component_sampling()
            else:
                e, g = self.circuit.grad_run()
        elif dense_mode:
            if component_sampling:
                e, g = self.circuit.sample_grad_observable_with_component_sampling(shot_num=shot_num)
            else:
                e, g = self.circuit.sample_grad_observable(shot_num=shot_num)
        else:
            if component_sampling:
                e, g = self.circuit.sample_grad_with_component_sampling(shot_num=shot_num)
            else:
                e, g = self.circuit.sample_grad(shot_num=shot_num)
        self.cost_history[self.iter] = e
        self.optimizer.step(g, e)
        self.iter += 1
        self.param_history[self.iter] = self.optimizer.parameters
        self.circuit.angles = self.optimizer.parameters

    def reset(self, **kwargs):
        ini_parameters = kwargs.get('ini_parameters', self.param_history[0])
        optimizer = kwargs.get('optimizer', self.optimizer_info)
        max_iter = kwargs.get('max_iter', self.max_iter)
        self.__init__(self.circuit, optimizer, max_iter, ini_parameters=ini_parameters)

class QaoaOpt(ParametrizedCircuitOptimizer):
    def __init__(self, circuit, optimizer, betas, gammas, max_iter=1000):
        ParametrizedCircuitOptimizer.init(self, circuit, max_iter)
        # set up history memory
        self.param_history = np.zeros([self.max_iter, circuit.lnum, 2], dtype='double')
        self.cost_history = np.zeros(self.max_iter, dtype='double')
        self.param_history[0] = np.array([betas, gammas]).transpose()
        # set up optimizer
        self.optimizer_info = optimizer
        self.pick(optimizer, np.array([betas, gammas]).transpose())

    def __str__(self):
        return self.optimizer_info.__str__()

    def step(self, shot_num=0, dense_mode=True):
        if self.iter >= self.max_iter:
            print('Maximum amount of iterations reached: {}.'.format(self.max_iter))
        if shot_num == 0:
            e, g = self.circuit.grad_run(self.param_history[self.iter, :, 0], self.param_history[self.iter, :, 1])
        elif dense_mode:
            e, g = self.circuit.sample_grad_dense(
                self.param_history[self.iter, :, 0],
                self.param_history[self.iter, :, 1],
                shot_num=shot_num
            )
        else:
            raise ValueError('dense_mode must be True, sparse sampling method is not implmented yet.')
        self.cost_history[self.iter] = e
        self.optimizer.step(g, e)
        self.iter += 1
        self.param_history[self.iter] = self.optimizer.parameters

    def reset(self, **kwargs):
        if ('betas' in kwargs) and ('gammas' in kwargs):
            self.param_history[0] = np.array([betas, gammas]).transpose()
        optimizer = kwargs.get('optimizer', self.optimizer_info)
        max_iter = kwargs.get('max_iter', self.max_iter)
        self.__init__(
            self.circuit,
            optimizer,
            self.param_history[0, :, 0],
            self.param_history[0, :, 1],
            max_iter
        )

### GRADIENT DESCENT METHODS
class GradientDescentOptimizer:
    def init(self, parameters, hyper_parameters):
        self.parameters = parameters
        self.step_size = hyper_parameters.get('step_size', .001)
        self.iter = 0

class Adam(GradientDescentOptimizer):
    def __init__(self, parameters, hyper_parameters):
        GradientDescentOptimizer.init(self, parameters, hyper_parameters)
        self.beta1 = hyper_parameters.get('beta1', .9)
        self.beta2 = hyper_parameters.get('beta2', .999)
        self.eps = hyper_parameters.get('eps', 10.**-8)
        self.m = np.zeros(parameters.shape, dtype='double')
        self.v = np.zeros(parameters.shape, dtype='double')
        self.m_hat = np.zeros(parameters.shape, dtype='double')
        self.v_hat = np.zeros(parameters.shape, dtype='double')
    def step(self, gradient, *args):
        self.iter += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * gradient
        self.v = self.beta2 * self.v + (1-self.beta2) * gradient**2
        self.m_hat = self.m / (1-self.beta1**self.iter)
        self.v_hat = self.v / (1-self.beta2**self.iter)
        self.parameters -= self.step_size * self.m_hat / (np.sqrt(self.v_hat) + self.eps)

class GradientDescent(GradientDescentOptimizer):
    def __init__(self, parameters, hyper_parameters):
        GradientDescentOptimizer.init(self, parameters, hyper_parameters)
        self.decay_function = hyper_parameters.get('decay_function', lambda step_size, iter: step_size)
    def step(self, gradient, *args):
        self.iter += 1
        self.parameters -= self.decay_function(self.step_size, self.iter) * gradient

class RateDecayOnPlateau(GradientDescentOptimizer):
    def __init__(self, parameters, hyper_parameters):
        GradientDescentOptimizer.init(self, parameters, hyper_parameters)
        self.plateau_length = hyper_parameters.get('plateau_length', 10)
        self.decay_rate = hyper_parameters.get('decay_rate', .5)
        self.plateau_counter = 0
        self.cost = 10.**10
    def step(self, gradient, new_cost):
        self.iter += 1
        if new_cost > self.cost:
            self.plateau_counter += 1
            if self.plateau_counter >= self.plateau_length:
                self.step_size *= self.decay_rate
                self.plateau_counter = 0
        else:
            self.cost = new_cost
            self.plateau_counter = 0
        self.parameters -= self.step_size * gradient
