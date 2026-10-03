import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

class Surrogate():
    @staticmethod
    def fast_sigmoid(u, v_th, gamma=10.0):
        sig = 1 / (1 + np.exp(-gamma*(u-v_th)))
        return gamma*sig*(1.0-sig)

class LIF():
    def __init__(self, num_inputs, num_neurons, beta=0.5, v_thres=1.0):
        self.beta = beta
        self.v_thres = v_thres

        #xavier/golort-like initialization for weights
        self.weights = np.random.randn(num_inputs, num_neurons) * np.sqrt(2.0/num_neurons)
        self.biases = np.zeros(num_neurons)

        #stored for back propagation
        self.cache = None

    def forward(self, input_spikes):
        """
        Executes the forward pass over T time steps.
        Args:
            input_spikes: Array of shape (num_steps, batch_size, num_inputs)
        Returns:
            spike_record: Output spikes of shape (num_steps, batch_size, num_neurons)
        """

        num_steps, batch_sizes, num_neurons = input_spikes.shape
        num_neurons = self.weights.shape[1]
        
        #initializing membrane potential for the batch
        u = np.zeros((batch_sizes, num_neurons))

        mem_record = []
        spike_record = []

        for t in range(num_steps):
            x_t = input_spikes[t]
            
            #synaptic input and membrane update
            synaptic_input = np.dot(x_t, self.weights) + self.biases
            u = self.beta * u + synaptic_input 

            mem_record.append(u.copy())
            
            #threshold/spike generation
            s_t = (u >= self.v_thres).astype(float)

            #soft reset
            u = u - s_t * self.v_thres

            spike_record.append(s_t.copy())

        self.cache = {
                "input_spikes":input_spikes,
                "mem_record":np.array(mem_record),
                "spike_record":np.array(spike_record)
            }
        if self.cache is None:
            raise RuntimeError("forward() must be called before backward()")

        return self.cache['spike_record']
        
    def backward(self, dL_dS, lr=0.01):
        """
        Executes Backpropagation Through Time (BPTT) using cached forward states.
        Args:
            dL_dS: Gradient of loss w.r.t output spikes, shape (num_steps, batch_size, num_neurons)
            lr: Learning rate for gradient descent update
        """
        if self.cache is None:
            raise RuntimeError("forward() must be called before backward()")

        input_spikes = self.cache['input_spikes']
        mem_record = self.cache['mem_record']
        spike_record = self.cache['spike_record']

        num_steps, batch_size, num_neurons = input_spikes.shape
        
        #initializing parameters
        dW = np.zeros_like(self.weights)
        dB = np.zeros_like(self.biases)

        # recurrent gredient tracking for future time step
        du_next = np.zeros_like(mem_record[0])

        for t in reversed(range(num_steps)):
            u_t = mem_record[t]
            s_t = input_spikes[t]
            
            dS_du = Surrogate.fast_sigmoid(u_t, self.v_thres)
            dL_du = (dL_dS[t]+du_next)*dS_du

            dB += np.sum(dL_du, axis=0)
            dW += np.dot(s_t.T, dL_du)

            du_next = dL_du * self.beta


        #prameter update via Gradient descent
        self.weights += - lr * dW
        self.biases += -lr * dB

# smoke testing of LIF
if __name__ == "__main__":
    np.random.seed(42)

    #dimentions of neurons 
    T = 25
    D_in = 2
    N_neurons = 2

    # X shape: (200, 2), y shape: (200,)
    X, y = spiral_data(samples=100, classes=2)

    batch_size = X.shape[0]

    # Convert coordinates to firing rates in [0.05, 0.8]
    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    firing_rates = 0.05 + 0.75 * X

    # Rate-code the 2D spiral points over T timesteps
    rng = np.random.default_rng(42)
    x_train = (
        rng.random((T, batch_size, D_in))
        < firing_rates[None, :, :]
    ).astype(float)

    y_target = np.zeros((T, batch_size, N_neurons))
    y_target[-1, np.arange(batch_size), y] = 1.0

    #initialization
    layer = LIF(num_inputs=D_in, num_neurons=N_neurons)

    print("Starting trianing looop : ")

    for epoch in range(10000):
        s_out = layer.forward(x_train)

        loss = np.mean((s_out[-1] - y_target[-1]) ** 2)

        dL_dS = np.zeros_like(s_out)
        dL_dS[-1] = (
            2 * (s_out[-1] - y_target[-1]) / batch_size
        )

        layer.backward(dL_dS, lr=0.1)

        if epoch % 100 == 0:
            predictions = np.argmax(s_out.sum(axis=0), axis=1)
            accuracy = np.mean(predictions == y)
            print(
                f"epoch: {epoch}, "
                f"loss: {loss:.4f}, "
                f"accuracy: {accuracy:.3f}"
            )