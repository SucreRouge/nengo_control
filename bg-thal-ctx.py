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
#
# def stim_input(t):
#     if (t // 6) % 2 == 0:
#         return 'ON'
#     else:
#         return 'OFF'
val1 = 14#2
val2 = 24#2.5
def stim_input(t):
    global val1
    global val2
    if (t <= val2) and (t >= val1):
        if t == val2:
            val1 += 24#2
            val2 += 24#2
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
    #actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    thalamus = nengo.Probe(model.thal.actions.output, synapse=0.01)
    bg = nengo.Probe(model.bg.gpi.output, synapse=0.01)
    #utility = nengo.Probe(model.bg.input, synapse=0.01)

############################
#### Run the Simulation ####
############################

# Create the simulator object
sim = nengo.Simulator(model)
# Simulate the model for 2 seconds
length_sim = 360 # 6 minute scan = 360
sim.run(length_sim)

ctx_output = model.similarity(sim.data, cortex)
thal_output = sim.data[thalamus]
bg_output = sim.data[bg]

########################################
#### Generate Hemodynamic Response #####
########################################

TR = 2
tr_time = np.arange(0, length_sim, TR)
hrf_at_trs = hrf(tr_time)

# sample from neural output
numNets = 3

neural_output = [[]] * numNets
convolved = [[]] * numNets
sampled_BOLD = []
for n in range(numNets):
    output = []
    if n == 0:
        on_output = ctx_output[:,0]
    elif n == 1:
        on_output = thal_output[:,0]
    elif n == 2:
        on_output = bg_output[:,0]

    for i in range(1,len(on_output)+1):
        if i % (TR * 1000) == 0:
            output.append(on_output[i-1])

    neural_output[n] = np.asarray(output)

    num_vols = len(neural_output[n])

    all_tr_times = np.arange(num_vols) * TR

    convolved[n] = np.convolve(neural_output[n], hrf_at_trs)

    remove = len(hrf_at_trs) - 1
    convolved[n] = convolved[n][:-remove]

    sampled = []
    for i in range(1,len(convolved[n])+1):
        #if i % (TR * 1000) == 0:
        sampled.append(convolved[n][i-1])

    sampled_BOLD.append(sampled)

#np.savetxt('simBOLD_360sec_TR2_14sOFF_10sON_stim.csv',np.asarray(sampled_BOLD),delimiter=',')

labels = ['ctx','thal','bg']

fig  = plt.figure(figsize=(12,8))

for i in range(len(convolved)):
    p1 = fig.add_subplot(2,1,1)
    p1.plot(all_tr_times, neural_output[i], label=labels[i])
    p1.set_ylabel('neural output')
    #p1.set_title(labels[i])

    p2 = fig.add_subplot(2,1,2)
    p2.plot(np.arange(len(sampled_BOLD[i])), sampled_BOLD[i], label=labels[i])
    p2.set_ylabel('estimated BOLD response (TR=2)')


p1.legend(labels, 'upper right')
p2.legend(labels, 'upper right')

plt.show()

###########################
#### Plot probed info #####
###########################

# fig = plt.figure(figsize=(12,8))
# p1 = fig.add_subplot(3,1,1)
#
# p1.plot(sim.trange(), model.similarity(sim.data, cortex))
# p1.legend(model.get_output_vocab('cortex').keys, fontsize='x-small')
# p1.set_ylabel('State')
#
# p2 = fig.add_subplot(3,1,2)
# p2.plot(sim.trange(), sim.data[actions])
# p2_legend_txt = [a.effect for a in model.bg.actions.actions]
# p2.legend(p2_legend_txt, fontsize='x-small')
# p2.set_ylabel('Action')
#
# p3 = fig.add_subplot(3,1,3)
# p3.plot(sim.trange(), sim.data[utility])
# p3_legend_txt = [a.condition for a in model.bg.actions.actions]
# p3.legend(p3_legend_txt, fontsize='x-small')
# p3.set_ylabel('Utility')
#
# fig.subplots_adjust(hspace=0.2)

#plt.show()
