from collections import OrderedDict

import numpy as np
from .ode_integration_interface_system import ODEIntegrationInterfaceSystem
import openmdao.api as om


class ODEIntegrationInterface(object):
    """
    Given a system class, create a callable object with the signature required by scipy.integrate.ode.

    Internally, this is accomplished by constructing an OpenMDAO problem using the ODE with
    a single node.  The interface populates the values of the time, states, and controls,
    and then calls `run_model()` on the problem.  The state rates generated by the ODE
    are then returned back to scipy ode, which continues the integration.

    Parameters
    ----------
    ode_class : class
        The ODEClass belonging to the phase being simulated.
    time_options : dict of {str: TimeOptionsDictionary}
        The time options for the phase being simulated.
    state_options : dict of {str: StateOptionsDictionary}
        The state options for the phase being simulated.
    control_options : dict of {str: ControlOptionsDictionary}
        The control options for the phase being simulated.
    polynomial_control_options : dict of {str: PolynomialControlOptionsDictionary}
        The polynomial control options for the phase being simulated.
    parameter_options : dict of {str: ParameterOptionsDictionary}
        The parameter options for the phase being simulated.
    ode_init_kwargs : dict
        Keyword argument dictionary passed to the ODE at initialization.
    reports : bool or None or str or Sequence
        The reports argument to be passed to the subproblems.  By default, no subproblem reports are generated.
    """
    def __init__(self, ode_class, time_options, state_options, control_options,
                 polynomial_control_options, parameter_options, ode_init_kwargs=None,
                 reports=False):

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = OrderedDict()
        self.time_options = time_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self.parameter_options = parameter_options
        self.control_interpolants = {}
        self.polynomial_control_interpolants = {}

        pos = 0

        for state, options in state_options.items():
            self.state_options[state] = {'rate_source': options['rate_source'],
                                         'pos': pos,
                                         'shape': options['shape'],
                                         'size': np.prod(options['shape']),
                                         'units': options['units'],
                                         'targets': options['targets']}
            pos += self.state_options[state]['size']

        self._state_vec = np.zeros(pos, dtype=float)
        self._state_rate_vec = np.zeros(pos, dtype=float)

        #
        # Build odeint problem interface
        #
        self.prob = om.Problem(model=ODEIntegrationInterfaceSystem(ode_class=ode_class,
                                                                   time_options=time_options,
                                                                   state_options=state_options,
                                                                   control_options=control_options,
                                                                   polynomial_control_options=polynomial_control_options,
                                                                   parameter_options=parameter_options,
                                                                   ode_init_kwargs=ode_init_kwargs),
                               reports=reports)

    def _unpack_state_vec(self, x):
        """
        Given the state vector in 1D, extract the values corresponding to
        each state into the ode integrators problem states.

        Parameters
        ----------
        x : np.array
            The 1D state vector.

        Returns
        -------
        None

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self.prob[f'states:{state_name}'][0, ...] = x[pos:pos + size]

    def _pack_state_rate_vec(self):
        """
        Pack the state rates into a 1D vector for use by scipy odeint.

        Returns
        -------
        dXdt : np.array
            The 1D state-rate vector.

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self._state_rate_vec[pos:pos + size] = \
                np.ravel(self.prob[f'state_rate_collector.state_rates:{state_name}_rate'])
        return self._state_rate_vec

    def set_interpolant(self, name, interp):
        """
        Set the control and/or polynomial control interpolants in the underlying system.

        Parameters
        ----------
        name : str
            The name of the control or polynomial control whose interpolant is being set.
        interp : LagrangeBarycentricInterpolant
            The LagrangeBarycentricInterpolant for the given control or polynomial control.
        """
        self.prob.model.set_interpolant(name, interp)

    def setup_interpolant(self, name, x0, xf, f_j):
        """
        Setup the values to be interpolated in an existing interpolant.

        Parameters
        ----------
        name : str
            The name of the control or polynomial control.
        x0 : float
            The initial time (or independent variable) of the segment (for controls) or phase (for polynomial controls).
        xf : float
            The final time (or independent variable) of the segment (for controls) or phase (for polynomial controls).
        f_j : float
            The value of the control at the nodes in the segment or phase.
        """
        if name in self.prob.model.options['control_options']:
            self.prob.model.setup_interpolant(name, x0, xf, f_j)
        elif name in self.prob.model.options['polynomial_control_options']:
            self.prob.model.setup_interpolant(name, x0, xf, f_j)
        else:
            raise KeyError(f'Unable to set control interpolant of unknown control: {name}')

    def __call__(self, t, x):
        """
        The function interface used by scipy.ode

        Parameters
        ----------
        t : float
            The current time, t.
        x : np.array
            The 1D state vector.

        Returns
        -------
        xdot : np.array
            The 1D vector of state time-derivatives.
        """
        self.prob['t'] = t
        self.prob['t_phase'] = t - self.prob['t_initial']
        self._unpack_state_vec(x)
        self.prob.run_model()
        xdot = self._pack_state_rate_vec()
        return xdot