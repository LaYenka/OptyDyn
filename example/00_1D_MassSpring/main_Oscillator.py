import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')  # disable plotting to the screen

from dymos.examples.oscillator.oscillator_ode import OscillatorODE

# Instantiate an OpenMDAO Problem instance.
prob = om.Problem()

# Instantiate a Dymos Trajectory and add it to the Problem model.
traj = dm.Trajectory()
prob.model.add_subsystem('traj', traj)

# Instantiate a Phase and add it to the Trajectory.
# Here the transcription is necessary but not particularly relevant.
phase = dm.Phase(ode_class=OscillatorODE, transcription=dm.Radau(num_segments=4))
traj.add_phase('phase0', phase)

# Tell Dymos the states to be propagated using the given ODE.
phase.add_state('v', rate_source='v_dot', targets=['v'], units='m/s')
phase.add_state('x', rate_source='v', targets=['x'], units='m')

# The spring constant, damping coefficient, and mass are inputs to the system
# that are constant throughout the phase.
phase.add_parameter('k', units='N/m', targets=['k'])
phase.add_parameter('c', units='N*s/m', targets=['c'])
phase.add_parameter('m', units='kg', targets=['m'])

# Setup the OpenMDAO problem
prob.setup()

# Assign values to the times and states
prob.set_val('traj.phase0.t_initial', 0.0)
prob.set_val('traj.phase0.t_duration', 15.0)

prob.set_val('traj.phase0.states:x', 10.0)
prob.set_val('traj.phase0.states:v', 0.0)

prob.set_val('traj.phase0.parameters:k', 1.0)
prob.set_val('traj.phase0.parameters:c', 0.5)
prob.set_val('traj.phase0.parameters:m', 1.0)

# Perform a single execution of the model (executing the model is required before simulation).
prob.run_model()

# Perform an explicit simulation of our ODE from the initial conditions.
sim_out = traj.simulate(times_per_seg=50)

# Plot the state values obtained from the phase timeseries objects in the simulation output.
t_sol = prob.get_val('traj.phase0.timeseries.time')
t_sim = sim_out.get_val('traj.phase0.timeseries.time')

states = ['x', 'v']
fig, axes = plt.subplots(len(states), 1)
for i, state in enumerate(states):
    sol = axes[i].plot(t_sol, prob.get_val(f'traj.phase0.timeseries.{state}'), 'o')
    sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.{state}'), '-')
    axes[i].set_ylabel(state)
axes[-1].set_xlabel('time (s)')
fig.legend((sol[0], sim[0]), ('solution', 'simulation'), loc='lower right', ncol=2)
plt.savefig('trajectory.png')
