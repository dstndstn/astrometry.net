#! /usr/bin/env python

scales=[10,5,4,3,2]

nodes = [
    ('hydra', [10]),
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
    ]


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
    print '''
    <replica-group id="RSolver-scale%i">
      <load-balancing type="round-robin"/>
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
#      <load-balancing type="adaptive" load-sample="5" n-replicas="2"/>

