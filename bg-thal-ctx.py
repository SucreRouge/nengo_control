import nengo, math, random, sys
import nengo.spa as spa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hrf import hrf

########################
#### Model Creation ####
########################

dimensions = 16 #number of dimensions for semantic pointers

model = spa.SPA()
with model:
    model.cortex = spa.Buffer(dimensions=dimensions)

    #action mapping
    actions = spa.Actions(
        'dot(cortex, ON) --> cortex = ON',
        'dot(cortex, OFF) --> cortex = OFF',
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)

##########################
#### Input Definition ####
##########################

def stim_input(t):
    if (t // 6) % 2 == 0:
        return 'ON'
    else:
        return 'OFF'

with model:
    model.input = spa.Input(cortex=stim_input)

############################
#### Probe Model Output ####
############################

with model:
    cortex = nengo.Probe(model.cortex.state.output, synapse=0.01)
    actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    utility = nengo.Probe(model.bg.input, synapse=0.01)

############################
#### Run the Simulation ####
############################

# Create the simulator object
sim = nengo.Simulator(model)
# Simulate the model for 2 seconds
length_sim = 360 # 6 minute scan = 360
sim.run(length_sim)

ctx_output = model.similarity(sim.data, cortex)

#################################
#### Generate BOLD Response #####
#################################

TR = 2
tr_time = np.arange(0, length_sim, TR)

hrf_at_trs = hrf(tr_time)

# sample from neural output
on_output = ctx_output[:,1]
output = []
for i in range(1,len(on_output)+1):
    #if i % (TR * 1000) == 0:
    if i % 400 == 0:
        output.append(on_output[i-1])
neural_output = np.asarray(output)
num_vols = len(neural_output)

all_tr_times = np.arange(num_vols) * TR

convolved = np.convolve(neural_output, hrf_at_trs)
remove = len(hrf_at_trs) - 1
convolved = convolved[:-remove]

plt.plot(all_tr_times, neural_output)
plt.plot(all_tr_times, convolved)
plt.show()

###########################
#### Plot probed info #####
###########################

fig = plt.figure(figsize=(12,8))
p1 = fig.add_subplot(3,1,1)

p1.plot(sim.trange(), model.similarity(sim.data, cortex))
p1.legend(model.get_output_vocab('cortex').keys, fontsize='x-small')
p1.set_ylabel('State')

p2 = fig.add_subplot(3,1,2)
p2.plot(sim.trange(), sim.data[actions])
p2_legend_txt = [a.effect for a in model.bg.actions.actions]
p2.legend(p2_legend_txt, fontsize='x-small')
p2.set_ylabel('Action')

p3 = fig.add_subplot(3,1,3)
p3.plot(sim.trange(), sim.data[utility])
p3_legend_txt = [a.condition for a in model.bg.actions.actions]
p3.legend(p3_legend_txt, fontsize='x-small')
p3.set_ylabel('Utility')

fig.subplots_adjust(hspace=0.2)

#plt.show()
