import nengo, math, random, sys
import nengo.spa as spa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from hrf import hrf

########################
#### Model Creation ####
########################

nSubs = 10
vects = {'input1':[],'input2':[],'input3':[]}
for subject in range(nSubs):
    r_1 = random.uniform(.9,1.0)
    r_12 = 1 - r_1
    r_2 = random.uniform(.35,.45)
    r_22 = 1 - (2*r_2)
    r_3 = random.uniform(.2,.3)
    r_32 = 1 - (3*r_3)

    vects['input1'].append([r_1,r_12,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    vects['input2'].append([r_2,r_22,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0])
    vects['input3'].append([0,0,0,0,0,0,0,0,0,0,0,0,r_3,r_32,r_3,r_3])

    dimensions = 16 #number of dimensions for semantic pointers

    # change these slightly to create individual variation
    vocab = spa.Vocabulary(dimensions, randomize=False)
    # vocab.add('ON_1',[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # vocab.add('ON_2',[0.4,0.4,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # vocab.add('REST',[0,0,0,0,0,0,0,0,0,0,0,0,0.25,0.25,0.25,0.25])
    vocab.add('ON_1',vects['input1'][subject])
    vocab.add('ON_2',vects['input2'][subject])
    vocab.add('REST',vects['input3'][subject])

    model = spa.SPA(dimensions,vocabs=vocab)
    with model:
        model.cortex = spa.Buffer(dimensions=dimensions)

        #print model.get_default_vocab(dimensions).parse("ON_1")

        #action mapping
        actions = spa.Actions(
            'dot(cortex, ON_1) --> cortex = ON_1',
            'dot(cortex, ON_2) --> cortex = ON_2',
            'dot(cortex, REST) --> cortex = REST',
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
    stim_length = 1
    rest_length = 1

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
                return "ON_2"


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

    filename = 'generated_data/simBOLD_TR07_2on_rest_stim_sub_%s.csv'%str(subject)
    np.savetxt(filename,np.asarray(sampled_BOLD),delimiter=',')

    filename = 'generated_data/rawOutput_TR07_2on_rest_stim_sub_%s.csv'%str(subject)
    np.savetxt(filename,np.asarray(neural_output),delimiter=',')

    # labels = ['ctx','thal','bg']
    #
    # fig  = plt.figure(figsize=(12,8))
    #
    # for i in range(len(convolved)):
    #     p1 = fig.add_subplot(2,1,1)
    #     p1.plot(all_tr_times, neural_output[i], label=labels[i])
    #     p1.set_ylabel('neural output')
    #     #p1.set_title(labels[i])
    #
    #     p2 = fig.add_subplot(2,1,2)
    #     p2.plot(np.arange(len(sampled_BOLD[i])), sampled_BOLD[i], label=labels[i])
    #     p2.set_ylabel('estimated BOLD response (TR=0.7)')
    #
    #
    # p1.legend(labels, 'upper right')
    # p2.legend(labels, 'upper right')
    #
    # plt.show()

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
