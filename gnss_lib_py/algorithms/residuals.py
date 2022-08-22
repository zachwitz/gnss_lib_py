"""Calculate residuals

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import numpy as np

from gnss_lib_py.parsers.navdata import NavData

def solve_residuals(measurements, comparison, inplace=True):
    """Calculates residuals given pseudoranges and receiver position.

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    comparison : gnss_lib_py.parsers.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters and the estimated or ground truth receiver clock bias
        also in meters as an instance of the NavData class with the
        following rows: x_*_m, y_*_m, z_*_m, b_*_m.
    inplace : bool
        If False, will return new NavData instance with gps_millis and
        reisuals. If True, will add a "residuals" rows in the
        current NavData instance.

    Returns
    -------
    new_navdata : gnss_lib_py.parsers.navdata.NavData or None
        If inplace is False, returns new NavData instance containing
        "gps_millis" and residual rows. If inplace is True, returns
        None.

    """

    # verify corrected pseudoranges exist in inputs
    measurements.in_rows(["gps_millis","corr_pr_m"])
    comparison.in_rows(["gps_millis"])

    # check for comparison indexes
    comp_idxs = {"x_*_m" : [],
                 "y_*_m" : [],
                 "z_*_m" : [],
                 "b_*_m" : [],
                 }
    for name, indexes in comp_idxs.items():
        indexes = [row for row in comparison.rows
                      if row.startswith(name.split("*",maxsplit=1)[0])
                       and row.endswith(name.split("*",maxsplit=1)[1])]
        if len(indexes) > 1:
            raise KeyError("Multiple possible row indexes for " \
                         + name \
                         + ". Unable to resolve for solve_wls.")
        if len(indexes) == 0:
            raise KeyError("Missing required " + name + " row for " \
                        + "solve_wls.")
        # must call dictionary to avoid pass by value
        comp_idxs[name] = indexes[0]

    residuals = []
    for timestamp, _, measurement_subset in measurements.loop_time("gps_millis"):

        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T

        num_svs = pos_sv_m.shape[0]

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # find time index for comparison NavData
        comp_t_idx = np.argmin(np.abs(comparison["gps_millis"] - timestamp))

        rx_pos = comparison[[comp_idxs["x_*_m"],
                             comp_idxs["y_*_m"],
                             comp_idxs["z_*_m"]],
                             comp_t_idx].reshape(-1,1)
        pos_rx_m = np.tile(rx_pos.T, (num_svs, 1))

        gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True) \
                + comparison[comp_idxs["b_*_m"],comp_t_idx]

        # calculate residual
        residuals_epoch = corr_pr_m - gt_pr_m

        residuals += np.squeeze(residuals_epoch).tolist()

    if inplace:
        # add measurements to measurement class
        measurements["residuals"] = residuals
        return None

    # if not inplace, create new NavData instance to return
    residual_navdata = NavData()
    residual_navdata["residuals"] = residuals
    residual_navdata["gps_millis"]= measurements["gps_millis"]
    for row in ["gnss_id","sv_id","signal_type"]:
        if row in measurements.rows:
            residual_navdata[row] = measurements[row]

    return residual_navdata
