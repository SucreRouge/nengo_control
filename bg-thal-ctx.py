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
val1 = 7#2
val2 = 12#2.5
def stim_input(t):
    global val1
    global val2
    if (t <= val2) and (t >= val1):
        if t == val2:
            val1 += 12#2
            val2 += 12#2
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

#################################
#### Generate BOLD Response #####
#################################

TR = 2
tr_time = np.arange(0, length_sim, TR)

hrf_at_trs = hrf(tr_time)

# sample from neural output
ctx_on_output = ctx_output[:,0] #1
thal_on_output = thal_output[:,0]
bg_on_output = bg_output[:,0]
output_ctx = []
output_thal = []
output_bg = []
for i in range(1,len(ctx_on_output)+1):
    if i % (TR * 1000) == 0:
    #if i % 200 == 0:
        output_ctx.append(ctx_on_output[i-1])
        output_thal.append(thal_on_output[i-1])
        output_bg.append(bg_on_output[i-1])

neural_output = [[]] * 3
neural_output[0] = np.asarray(output_ctx)
neural_output[1] = np.asarray(output_thal)
neural_output[2] = np.asarray(output_bg)

num_vols = len(neural_output[0])

all_tr_times = np.arange(num_vols) * TR

convolved = [[]] * 3
convolved[0] = np.convolve(neural_output[0], hrf_at_trs)
convolved[1] = np.convolve(neural_output[1], hrf_at_trs)
convolved[2] = np.convolve(neural_output[2], hrf_at_trs)

remove = len(hrf_at_trs) - 1
convolved[0] = convolved[0][:-remove]
convolved[1] = convolved[1][:-remove]
convolved[2] = convolved[2][:-remove]

np.savetxt('simBOLD_360sec_TR2_7sOFF_5sON_stim.csv',np.asarray(convolved),delimiter=',')

fig = plt.figure(figsize=(12,8))
ylabels = ['ctx','thal','bg']
for i in range(len(convolved)):
    p = fig.add_subplot(3,1,i)
    p.plot(all_tr_times, neural_output[i])
    p.plot(all_tr_times, convolved[i])
    p.set_ylabel(ylabels[i])

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
