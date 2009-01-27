#! /usr/bin/env python

# Note, you'll also want to edit "startallnodes" to actually start up the icegridnode daemons
# on the appropriate machines.

scales=[10,5,4,3,2]

#hosts = ['neuron-0-%i' for i in range(25)]
all_nodes = [('neuron-0-%i' % i, [scales[i % len(scales)]]) for i in range(25)]

all_nodes_OLD = [
    ('neuron-0-0', [10, 5]),
    ('neuron-0-1', [4]),
    ('neuron-0-2', [3]),
    ('neuron-0-3', [2]),
    ('neuron-0-4', [10, 5]),
    ('neuron-0-5', [4]),
    ('neuron-0-6', [3]),
    ('neuron-0-7', [2]),
    ('neuron-0-8', [10, 5]),
    ('neuron-0-9', [4]),
    ('neuron-0-10', [3]),
    ('neuron-0-11', [2]),
    ('neuron-0-12', [10, 5]),
    ('neuron-0-13', [4]),
    ('neuron-0-14', [3]),
    ('neuron-0-15', [2]),
    ('neuron-0-16', [10, 5]),
    ('neuron-0-17', [4]),
    ('neuron-0-18', [3]),
    ('neuron-0-19', [2]),
    ('neuron-0-20', [10, 5]),
    ('neuron-0-21', [4]),
    ('neuron-0-22', [3]),
    ('neuron-0-23', [2]),
    ]

some_nodes = [
    ('neuron-0-0', [10]),
    ('neuron-0-1', [5]),
    ('neuron-0-2', [4]),
    ('neuron-0-3', [3]),
    ('neuron-0-4', [2]),
    ('neuron-0-5', [10]),
    ('neuron-0-6', [5]),
    ('neuron-0-7', [4]),
    ('neuron-0-8', [3]),
    ('neuron-0-9', [2]),
    ('neuron-0-10', [10]),
    ('neuron-0-11', [5]),
    ]

TEST_nodes = [
    ('neuron-0-0', [10]),
    ('neuron-0-1', [5]),
    ('neuron-0-2', [4]),
    ('neuron-0-3', [3]),
    ('neuron-0-4', [2]),
    ('neuron-0-5', [10]),
    ]

nodes = all_nodes

print '''
<icegrid>
  <application name="Solver">

'''

# <server
#              activation="on-demand">
#              activation="always">


for scale in scales:
    print '''
    <server-template id="ServerTemplate-scale%i">
      <parameter name="index"/>
      <server id="Server-${index}-scale%i"
              exe="python"
              activation="always">
        <option>SolverServer.py</option>
        <option>%i</option>
        <adapter name="OneSolver" endpoints="tcp"
         replica-group="RSolver-scale%i"/>
        <property name="Identity" value="Solver-scale%i"/>
        <property name="Ice.ThreadPool.Server.SizeMax" value="10"/>
        </server>
    </server-template>
''' % ((scale,)*5)

for scale in scales:
    #<load-balancing type="round-robin"/>
    print '''
    <replica-group id="RSolver-scale%i">
    <load-balancing type="adaptive"/>
      <object identity="Solver-scale%i" type="::SolverIce::Solver"/>
    </replica-group>
    ''' % ((scale,)*2)

for (node, scales) in nodes:
    print '<node name="%s">' % node
    for scale in scales:
        print ('  <server-instance template="ServerTemplate-scale%i" index="%s"/>'
               % (scale, node))
    print '</node>'

print '''
  </application>

</icegrid>
'''

