"""Functions to process Android measurements.

"""

__authors__ = "Shubh Gupta, Adam Dai, Ashwin Kanhere"
__date__ = "02 Nov 2021"

import os
import sys
import csv

import numpy as np
import pandas as pd
# append <path>/gnss_lib_py/gnss_lib_py/ to path

sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
from parsers.measurement import Measurement


class AndroidDerived(Measurement):
    """Class handling derived measurements from Android dataset.

    Inherits from Measurement().
    """
    #NOTE: Inherits __init__() and isn't defined explicitly here because
    # no additional implementations/attributes are needed

    def preprocess(self, input_path):
        """Android specific loading and preprocessing for Measurement()

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names
        """
        pd_df = pd.read_csv(input_path)
        col_map = self._column_map()
        pd_df.rename(columns=col_map, inplace=True)
        return pd_df

    def postprocess(self):
        """Android derived specific postprocessing for Measurement()

        Notes
        -----
        Adds corrected pseudoranges to measurements. Corrections
        implemented from https://www.kaggle.com/carlmcbrideellis/google-smartphone-decimeter-eda
        retrieved on 9th November, 2021
        """
        pr_corrected = self['rawPrM', :] + self['b_sat_m', :] - \
                    self['isrbM', :] - self['tropoDelayM', :] \
                    - self['ionoDelayM', :]
        self['pseudo'] = pr_corrected

    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'millisSinceGpsEpoch' : 'toeMillis',
                'svid' : 'PRN',
                'xSatPosM' : 'x_sat_m',
                'ySatPosM' : 'y_sat_m',
                'zSatPosM' : 'z_sat_m',
                'xSatVelMps' : 'vx_sat_mps',
                'ySatVelMps' : 'vy_sat_mps',
                'zSatVelMps' : 'vz_sat_mps',
                'satClkBiasM' : 'b_sat_m',
                'satClkDriftMps' : 'b_dot_sat_mps',
                }
        return col_map


class AndroidRawImu(Measurement):
    """Class handling IMU measurements from raw Android dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path, group_time=10):
        self.group_time = group_time
        data_df = self.preprocess(input_path)
        self.build_measurement(data_df)
        self.postprocess()

    def preprocess(self, input_path):
        """Read Android raw file and produce IMU dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.
        Returns
        -------
        accel : pd.DataFrame
            Dataframe that contains the accel measurements from the log.
        gyro : pd.DataFrame
            Dataframe that contains the gyro measurements from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Accel' in row[0]:
                        accel = [row[1:]]
                    elif 'Gyro' in row[0]:
                        gyro = [row[1:]]
                else:
                    if row[0] == 'Accel':
                        accel.append(row[1:])
                    elif row[0] == 'Gyro':
                        gyro.append(row[1:])

        accel = pd.DataFrame(accel[1:], columns = accel[0], dtype=np.float64)
        gyro = pd.DataFrame(gyro[1:], columns = gyro[0], dtype=np.float64)

        #Drop common columns from gyro and keep values from accel
        gyro.drop(columns=['utcTimeMillis', 'elapsedRealtimeNanos'], inplace=True)
        measurements = pd.concat([accel, gyro], axis=1)
        #NOTE: Assuming pandas index corresponds to measurements order
        measurements.rename(columns=self._column_map(), inplace=True)
        return measurements

    @staticmethod
    def _column_map():
        col_map = {'AccelXMps2' : 'acc_x_mps2',
                'AccelYMps2' : 'acc_y_mps2',
                'AccelZMps2' : 'acc_z_mps2',
                'GyroXRadPerSec' : 'omega_x_radps',
                'GyroYRadPerSec' : 'omega_y_radps',
                'GyroZRadPerSec' : 'omega_z_radps',
                }
        return col_map


class AndroidRawFixes(Measurement):
    """Class handling location fix measurements from raw Android dataset.

    Inherits from Measurement().
    """
    def preprocess(self, input_path):
        """Read Android raw file and produce location fix dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.
        Returns
        -------
        fix_df : pd.DataFrame
            Dataframe that contains the location fixes from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes = [row[1:]]
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])

        fix_df = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
        return fix_df


def make_csv(input_path, field):
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    fields : list of strings
        Type of data to extract. Valid options are either "Raw",
        "Accel", "Gyro", "Mag", or "Fix".

    Returns
    -------
    out_path : string
        New file location of the exported CSV.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with MakeCsv() in opensource/ReadGnssLogger.m

    """
    out_path = field + ".csv"
    with open(out_path, 'w', encoding="utf8") as out_csv:
        writer = csv.writer(out_csv)
        with open(input_path, 'r', encoding="utf8") as in_txt:
            for line in in_txt:
                # Comments in the log file
                if line[0] == '#':
                    # Remove initial '#', spaces, trailing newline
                    # and split using commas as delimiter
                    line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
                # Data in file
                else:
                    # Remove spaces, trailing newline and split using commas as delimiter
                    line_data = line.rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
    return out_path
