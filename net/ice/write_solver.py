#! /usr/bin/env python

scales=[10,5,4]

print '''
<icegrid>
  <application name="Solver">

'''

# <server              activation="on-demand">


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
    
print '''
    <node name="hydra">
'''
for scale in scales:
    print '''
      <server-instance template="ServerTemplate-scale%i" index="hydra"/>
      ''' % ((scale,)*1)
print '''
    </node>

  </application>

</icegrid>
'''
#      <load-balancing type="adaptive" load-sample="5" n-replicas="2"/>

