# MDA Gemseo
#
from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_mda

configure_logger()

def display_result(res, mda_name):
    """Display coupling and output variables in logger.

    @param res: result (dict) of MDA
    @param mda_name: name of the current MDA
    """
    # names of the coupling variables
    coupling_names = [
        "y_11",
        "y_12",
        "y_14",
        "y_21",
        "y_23",
        "y_24",
        "y_31",
        "y_32",
        "y_34",
    ]
    for coupling_var in coupling_names:
        print(
            "{}, coupling variable {}: {}".format(
                mda_name, coupling_var, res[coupling_var]
            ),
        )

    # names of the output variables
    output_names = ["y_1", "y_2", "y_3", "y_4", "g_1", "g_2", "g_3"]
    for output_name in output_names:
        print(
            f"{mda_name}, output variable {output_name}: {res[output_name]}",
        )


disciplines = create_discipline(
    [
        "SobieskiStructure",
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
    ]
)
mda = create_mda("MDAGaussSeidel", disciplines)
res = mda.execute()
display_result(res, mda.name)
mda.plot_residual_history(n_iterations=10, logscale=[1e-8, 10.0], save=True, show=True)
