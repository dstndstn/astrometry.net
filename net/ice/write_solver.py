#! /usr/bin/env python

# Note, you'll also want to edit "startallnodes" to actually start up the icegridnode daemons
# on the appropriate machines.

scales=[10,5,4,3,2]

### TEMP - avoid nodes 12-15. (until Mar 11/2009)
#nodes = [('neuron-0-%i' % i, [scales[i % len(scales)]]) for i in range(25)]
#hosts = ['neuron-0-%i' for i in range(25)]

hosts = ['neuron-0-%i' % i for i in (range(12) + range(16, 25))]
nodes = [(h, [scales[i % len(scales)]]) for i,h in enumerate(hosts)]

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
        <property name="Ice.ThreadPool.Server.SizeMax" value="20"/>
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

