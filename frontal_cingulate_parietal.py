import nengo
import nengo.spa as spa

D = 32  # the dimensionality of the vectors

model = nengo.Network()
with model:
    GO = nengo.Node([0])
    STOP = nengo.Node([0])

    ctx = nengo.Network()
    with ctx:
        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=1)
        c = nengo.Ensemble(n_neurons=300, dimensions=2)

        nengo.Connection(a, c[0])
        nengo.Connection(b, c[1])
    nengo.Connection(GO, a)
    nengo.Connection(STOP, b)
