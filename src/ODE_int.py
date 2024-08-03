import numpy as np

class ode_int():
    """This class implements integral numerical methods to solve PDE. At the moment, euler and rk4 are available as methods

    Returns:
        _type_: _description_
    """
    def euler(self,f, y0, t,args):
        ##### Euler's Method #####
        """_summary_

        Args:
            f (_type_): _description_
            y0 (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        y = np.zeros(len(t))
        y[0] = np.asarray(y0)
        for i in range(0, len(t) - 1):
            y[i + 1] = y[i] + f(y[i], t[i]) * (t[i + 1] - t[i],args)
        return y

    def rk4(self,f, y0, t, *args):
        ##### Runge Kutta 4th Order Method #####
        """_summary_

        Args:
            f (_type_): _description_
            y0 (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        # define numpy array vector of the function
        y = np.zeros([len(y0),len(t)],dtype=float)
        y[:,0] = np.asarray(y0)
        for i in range(0, len(t) - 1):
            h = t[i + 1] - t[i]
            F1 = h * f(y[i], t[i],args)
            F2 = h * f((y[i] + F1 / 2), (t[i] + h / 2),args)
            F3 = h * f((y[i] + F2 / 2), (t[i] + h / 2),args)
            F4 = h * f((y[i] + F3), (t[i] + h),args)
            y[i + 1] = y[i] + 1 / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
        return y
