import numpy as np
import matplotlib.pyplot as pyplot
import nengo

'''
Learning...
This is basically just the basal ganglia example
'''

# Create the network (BG and action input node)

net = nengo.Network(label='Basal Ganglia')

with model:
    basal_ganglia = nango.networks.BasalGanglia(dimensions=3)

class ActionIterator(object):
    def __init__(self, dimensions):
        self.actions = np.ones(dimensions) * 0.1

    def step(self, t):
        dominate = int(t % 3)
        self.actions[:] = 0.1
        self.actions[dominate] = 0.8
        return self.actions

action_iterator = ActionIterator(dimensions=3)

with model:
    actions = nengo.Node(action_iterator.step, label="actions")

# Connect the network

with model:
    nengo.Connection(actions, basal_ganglia.input, synapse=None)
    selected_action = nengo.Probe(basal_ganglia.output, synapse=0.01)
    input_actions = nengo.Probe(actions, synapse=0.01)

# Simulate the Network

sim = nengo.Simulator(model)

sim.run(6)

# Plot Results
