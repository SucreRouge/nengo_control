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
        'dot(cortex, ON_1) --> cortex = ON_1',
        'dot(cortex, ON_2) --> cortex = ON_2',
        'dot(cortex, REST) --> cortex = REST'
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)

##########################
#### Input Definition ####
##########################
#

trialCount = 0
elapsed = 0
nTrials = 4
stim_length = 20
rest_length = 40

def trialLength():
    global stim_length
    #times = [0.2,0.3,0.4,0.5,0.6]
    times = [stim_length]
    return random.choice(times)

def trialType(i):
    if i % 2 == 0:
        return "ON_1"
    else:
        return "ON_2"
    #return random.choice(["ON_1","ON_2"])

def ITI():
    global rest_length

    times = [0.5,0.5,0.5,0.5]
    times = [rest_length]
    return random.choice(times)

trial_params = {"len":[],"iti":[],"type":[]}
for i in range(nTrials):
    trial_params["len"].append(trialLength())
    trial_params["iti"].append(ITI())
    trial_params["type"].append(trialType(i))

def stim_input(t):
    global trialCount
    global elapsed
    global trial_params

    if trialCount == nTrials:
        t_ind = trialCount - 1
    else:
        t_ind = trialCount

    trial_length = trial_params["len"][t_ind]
    iti = trial_params["iti"][t_ind]
    trial_type = trial_params["type"][t_ind]

    if t <= iti + elapsed:

        return "REST"

    elif t > iti + elapsed and t <= iti + elapsed + trial_length:
        ceiling = iti + elapsed + trial_length
        if t == ceiling or t > (ceiling - 0.005):
            trialCount += 1
            elapsed = t
        if trial_type == "ON_1":
            return "ON_1"
        else:
            return "ON_1*ON_2"


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
#length_sim = 10 # 6 minute scan = 360
length_sim = np.sum(trial_params["len"]) + np.sum(trial_params["iti"])
sim.run(length_sim)

ctx_output = model.similarity(sim.data, cortex)
thal_output = sim.data[thalamus]
bg_output = sim.data[bg]

########################################
#### Generate Hemodynamic Response #####
########################################

TR = 0.7
tr_time = np.arange(0, length_sim, TR)
hrf_at_trs = hrf(tr_time)

# sample from neural output
numNets = 3

neural_output = [[]] * numNets
convolved = [[]] * numNets
sampled_BOLD = []

mean_ctx = np.mean(ctx_output,axis=1)
mean_thal = np.mean(thal_output,axis=1)
mean_bg = np.mean(bg_output,axis=1)

for n in range(numNets):
    output = []
    if n == 0:
        #on_output = ctx_output[:,0]
        on_output = mean_ctx
    elif n == 1:
        #on_output = thal_output[:,0]
        on_output = mean_thal
    elif n == 2:
        #on_output = bg_output[:,0]
        on_output = mean_bg

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
