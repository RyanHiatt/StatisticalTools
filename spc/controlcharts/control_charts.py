import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
from scipy import stats


"""
Control Chart Module
Version 0.1.0
Created by: Ryan Hiatt
Last Updated: 04/19/2022
"""


class ControlChart:

    def __init__(self, data, chart_path: str = None):

        self.data = self.standardize_data(data)
        self.n = len(self.data.columns)
        self.datatype = self.check_datatype(self.data)

        self.chart_path = chart_path

        # Control Chart Variables
        self.control_chart_factors = pd.read_csv(
            'control_chart_factors.csv', index_col='n')

    """
    Chart types:

        Continuous:
            X Chart (n = 1)
            mR Chart (n = 1)
            X and mR Charts (n = 1)

            "range" X-bar Chart (1 < n < 11)
            R Chart (1 < n < 11)
            X-bar and R Charts (1 < n < 11)

            "stddev" X-bar Chart (n > 10)
            S Chart (n > 10)
            X-bar and S Chart (n > 10)

        Discrete:
            C Chart (Multiple defects per unit, constant sample size)
            U Chart (Multiple defects per unit, un-constant sample size)
            np Chart (Single defect per unit, constant sample size)
            p Chart (Single defect per unit, un-constant sample size)

    Possibly Include:
        Process Capability

    """

    @staticmethod
    def standardize_data(data):
        formatted_data = pd.DataFrame(data)
        formatted_data.columns = [
            f'x{y+1}' for y in range(len(formatted_data.columns))]
        return formatted_data

    @staticmethod
    def check_datatype(data):
        if np.array_equal(data.values, data.values.astype(int)):
            datatype = {'dist': 'discrete', 'type': 'int64'}
        else:
            datatype = {'dist': 'continuous', 'type': 'float64'}
        return datatype

    def data_summary(self):
        return self.data.describe()

    @staticmethod
    def validate_chart_points(name, points, ucl, lcl):
        print(f'{name} Validation:')
        control = True
        for i in range(len(points)):
            if points[i] > ucl or points[i] < lcl:
                control = False
                print(
                    f'Point {i} is out of control with a value of {points.loc[i]}')
        if control:
            print('All points in control.')

    @staticmethod
    def validate_chart_points_unequal_samples(name, points, ucl, lcl):
        print(f'{name} Validation:')
        control = True
        for i in range(len(points)):
            if points[i] > ucl[i] or points[i] < lcl[i]:
                control = False
                print(
                    f'Point {i} is out of control with a value of {points[i]}')
        if control:
            print('All points in control.')

    def single_plot(self, title, xlab, ylab, data, ucl, cl, lcl):
        # Create Single Chart
        fig, ax = plt.subplots(1, 1, sharex=True, squeeze=True,
                               figsize=(10, 3.5), tight_layout=True)

        ax.plot(data, linestyle='-', marker='o',
                color='slategray')  # Plot Data
        ax.axhline(ucl, linestyle='--', color='red',
                   label=f'UCL = {ucl:.2f}')  # UCL
        ax.axhline(cl, color='blue', label=f'CL = {cl:.2f}')  # CL
        ax.axhline(lcl, linestyle='--', color='red',
                   label=f'LCL = {lcl:.2f}')  # LCL
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                  borderaxespad=0.)  # Plot Legend
        ax.set(title=title, xlabel=xlab, ylabel=ylab)  # Titles and Labels

        if self.chart_path:
            plt.savefig(f"{self.chart_path}/{title}.png")
        else:
            return plt.show()

    def single_plot_unequal_samples(self, title, xlab, ylab, data, ucl, cl, lcl):
        # Create Single Chart
        fig, ax = plt.subplots(1, 1, sharex=True, squeeze=True,
                               figsize=(10, 3.5), tight_layout=True)

        ax.plot(data, linestyle='-', marker='o',
                color='slategray')  # Plot Data
        ax.step(x=range(0, len(data)), y=ucl, linestyle='--', color='red',
                label='UCL')  # UCL
        ax.axhline(cl, color='blue', label=f'CL = {cl:.2f}')  # CL
        ax.step(x=range(0, len(data)), y=lcl, linestyle='--', color='red',
                label='LCL')  # LCL
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                  borderaxespad=0.)  # Plot Legend
        ax.set(title=title, xlabel=xlab, ylabel=ylab)  # Titles and Labels

        if self.chart_path:
            plt.savefig(f"{self.chart_path}/{title}.png")
        else:
            return plt.show()

    def double_plot(self, title, xlab, ylab, data, ucl, cl, lcl):
        # Create Double Chart
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True,
                                       figsize=(10, 7), tight_layout=True)

        # Plot One
        ax1.axhline(ucl[0], linestyle='--', color='red',
                    label=f'UCL = {ucl[0]:.2f}')  # UCL
        ax1.axhline(cl[0], color='blue', label=f'CL = {cl[0]:.2f}')  # CL
        ax1.axhline(lcl[0], linestyle='--', color='red',
                    label=f'LCL = {lcl[0]:.2f}')  # LCL

        ax1.plot(data[0], linestyle='-', marker='o',
                 color='black')  # Plot Data
        ax1.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                   borderaxespad=0.)  # Plot Legend
        ax1.set(title=title[0], xlabel=xlab[0],
                ylabel=ylab[0])  # Titles and Labels

        # Plot Two
        ax2.axhline(ucl[1], linestyle='--', color='red',
                    label=f'UCL = {ucl[1]:.2f}')  # UCL
        ax2.axhline(cl[1], color='blue', label=f'CL = {cl[1]:.2f}')  # CL
        ax2.axhline(lcl[1], linestyle='--', color='red',
                    label=f'LCL = {lcl[1]:.2f}')  # LCL

        ax2.plot(data[1], linestyle='-', marker='o',
                 color='black')  # Plot Data
        ax2.legend(bbox_to_anchor=(1.05, 0.5),
                   loc='center left', borderaxespad=0.)
        ax2.set(title=title[1], xlabel=xlab[1],
                ylabel=ylab[1])  # Titles and Labels

        if self.chart_path:
            plt.savefig(f"{self.chart_path}/{title}.png")
        else:
            return plt.show()

    # def auto_plot_chart(self):
    #     if self.datatype == 'continuous':
    #         print('continuous plot coming up')
    #
    #     elif self.datatype == 'discrete':
    #         print('discrete plot coming up')
    #
    #     else:
    #         raise ValueError()

    # X Chart -----------------------------------------------------------------
    def x_chart(self):
        """
        The X chart, also known as the individual chart, is used to monitor the
        mean and variation of a process based on individual samples taken in a
        given time. The y-axis shows the mean and the control limits while the
        x-axis shows the sample units.

        Criteria:
            Data type: Continuous
            Sample Size: n = 1

        Data Input:
            A single vector or a single dataframe columns of samples
        """
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('X chart must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item()
                             - self.data.iloc[i - 1].item()))

        # Data for Plot
        chart_data = self.data.assign(mR=mr.values)

        # Factors
        d2 = self.control_chart_factors.loc[2, 'd2']

        # Control Limits
        cl = statistics.mean(chart_data['x1'])
        ucl = cl + 3 * \
            statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2
        lcl = cl - 3 * \
            statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2

        # Validate that points are in control
        self.validate_chart_points('X Chart', chart_data['x1'], ucl, lcl)

        # Plot the data
        self.single_plot('X Chart', 'Sample #', 'Data Values',
                         chart_data['x1'], ucl, cl, lcl)

    # mR Chart ----------------------------------------------------------------
    def mR_chart(self):
        """
        The mR chart is used to monitor the mean and variation of a process
        based on individual samples taken in a given time. The y-axis shows the
        moving range grand mean and the control limits while the x-axis shows
        the sample units

        Criteria:
            Data type: Continuous
            Sample Size: n = 1

        Data Input:
            A single vector or a single dataframe columns of samples
        """
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('mR chart must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item()
                             - self.data.iloc[i - 1].item()))

        # Data for Plot
        chart_data = self.data.assign(mR=mr.values)

        # Factors
        D3 = self.control_chart_factors.loc[self.n + 1, 'D3']
        D4 = self.control_chart_factors.loc[self.n + 1, 'D4']

        # Control Limits
        cl = statistics.mean(chart_data['mR'][1:len(chart_data['mR'])])
        ucl = D4 * cl
        lcl = D3 * cl

        # Validate that points are in control
        self.validate_chart_points('mR Chart', chart_data['mR'], ucl, lcl)

        # Plot the data
        self.single_plot('mR Chart', 'Sample #', 'mR Values',
                         chart_data['mR'], ucl, cl, lcl)

        plt.show()

    # X & mR Chart ------------------------------------------------------------
    def x_mR_chart(self):
        """
        The x chart (also known as individual chart) and mR chart are used to
        monitor the mean and variation of a process based on individual
        samples taken in a given time. In order to use the mR chart along with
        the x chart, the sample size n must be equal to 1. On the x chart, the
        y-axis shows the mean and the control limits while the x-axis shows the
        sample units. On the mR chart, the y-axis shows the moving range grand
        mean and the control limits while the x-axis shows the sample units.

        Criteria:
            Data type: Continuous
            Sample Size: n = 1

        Data Input:
            A single vector or a single dataframe columns of samples
        """
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('X & mR charts must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item()
                             - self.data.iloc[i - 1].item()))

        # Data for Plot
        chart_data = self.data.assign(mR=mr.values)

        # Factors
        d2 = self.control_chart_factors.loc[self.n + 1, 'd2']
        D3 = self.control_chart_factors.loc[self.n + 1, 'D3']
        D4 = self.control_chart_factors.loc[self.n + 1, 'D4']

        # X Control Limits
        cl1 = statistics.mean(chart_data['x1'])
        ucl1 = cl1 + 3 * \
            statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2
        lcl1 = cl1 - 3 * \
            statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2

        # mR Control Limits
        cl2 = statistics.mean(chart_data['mR'][1:len(chart_data['mR'])])
        ucl2 = cl2 * D4
        lcl2 = cl2 * D3

        # Validate that points are in control
        self.validate_chart_points('X Chart', chart_data['x1'], ucl1, lcl1)
        print()
        self.validate_chart_points('mR Chart', chart_data['mR'], ucl2, lcl2)

        # Plot the data
        self.double_plot(('X Chart', 'mR Chart'), (None, 'Sample #'),
                         ('Data Values', 'mR Values'),
                         (chart_data['x1'], chart_data['mR']),
                         (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

    # X-bar (Range) Chart -----------------------------------------------------
    def xbarr_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        """
        The x-bar "range" chart is used to monitor the mean and variation of a
        process based on samples taken in a given time. On the x-bar chart,
        the y-axis shows the grand mean and the control limits while the x-axis
        shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: 1 < n < 11

        Data Input:
            Between 2 and 10 vectors or 2 to 10 dataframe columns
            (one row = one sample)
        """
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError(
                'X-bar(Range) chart must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(
            R=abs(self.data.max(1) - self.data.min(1)))

        # Factors
        A2 = self.control_chart_factors.loc[self.n, 'A2']

        if mu and not sigma or sigma and not mu:
            raise ValueError('Provide both mu and sigma')

        elif mu and sigma:  # Mu and Sigma known
            if limits:  # Sigma limits given
                # Control Limits
                cl = mu
                ucl = mu + limits * sigma / math.sqrt(self.n)
                lcl = mu - limits * sigma / math.sqrt(self.n)

                # Validate that points are in control
                self.validate_chart_points('X-bar(Range) Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = mu
                ucl = mu + stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))
                lcl = mu - stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))

                # Validate that points are in control
                self.validate_chart_points(f'X-bar(Range) Chart - Mu and Sigma Known - Standard Limits - Alpha={alpha}',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

        else:  # Mu and Sigma unknown
            # Control Limits
            cl = statistics.mean(chart_data['x_bar'])
            ucl = cl + A2 * statistics.mean(chart_data['R'])
            lcl = cl - A2 * statistics.mean(chart_data['R'])

            # Validate that points are in control
            self.validate_chart_points('X-bar(Range) Chart - Mu and Sigma Unknown - Standard Limits',
                                       chart_data['x_bar'], ucl, lcl)

            # Plot the data
            self.single_plot('X-bar Chart', 'Sample #',
                             'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

    # R Chart -----------------------------------------------------------------
    def R_chart(self, sigma=None, limits=None):
        """
        The R-chart is used to monitor the mean and variation of a
        process based on samples taken in a given time. On the R chart, the
        y-axis shows the range grand mean and the control limits while the
        x-axis shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: 1 < n < 11

        Data Input:
            Between 2 and 10 vectors or 2 to 10 dataframe columns
            (one row = one sample)
        """
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError('R chart must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(
            R=abs(self.data.max(1) - self.data.min(1)))

        # Factors
        d2 = self.control_chart_factors.loc[self.n, 'd2']
        d3 = self.control_chart_factors.loc[self.n, 'd3']
        D1 = self.control_chart_factors.loc[self.n, 'D1']
        D2 = self.control_chart_factors.loc[self.n, 'D2']
        D3 = self.control_chart_factors.loc[self.n, 'D3']
        D4 = self.control_chart_factors.loc[self.n, 'D4']

        if sigma:  # Sigma known
            if limits:  # Sigma limits given
                # Control Limits
                cl = d2 * sigma
                ucl = cl + limits * d3 * sigma
                lcl = cl - limits * d3 * sigma

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Known - Limits Given',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #',
                                 'R Values', chart_data['R'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = d2 * sigma
                ucl = D2 * sigma
                lcl = D1 * sigma

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Known - Standard Limits',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #',
                                 'R Values', chart_data['R'], ucl, cl, lcl)

        else:  # Sigma unknown
            if limits:  # Sigma limits given
                # Control Limits
                cl = statistics.mean(chart_data['R'])
                ucl = cl * (d2 + limits * d3) / d2
                lcl = cl * (d2 - limits * d3) / d2

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Unknown - Limits Given',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #',
                                 'R Values', chart_data['R'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = statistics.mean(chart_data['R'])
                ucl = cl * D4
                lcl = cl * D3

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Unknown - Standard Limits',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #',
                                 'R Values', chart_data['R'], ucl, cl, lcl)

    # X-bar & R Chart ---------------------------------------------------------
    def xbar_R_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        """
        The x-bar and R-chart are used to monitor the mean and variation of a
        process based on samples taken in a given time. In order to use the R
        chart along with the x-bar chart, the sample size n must be greater
        than 1 and less than 11.On the x-bar chart, the y-axis shows the grand
        mean and the control limits while the x-axis shows the sample group.
        On the R chart, the y-axis shows the range grand mean and the control
        limits while the x-axis shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: 1 < n < 11

        Data Input:
            Between 2 and 10 vectors or 2 to 10 dataframe columns
            (one row = one sample)
        """
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError(
                'X-bar & R charts must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(
            R=abs(self.data.max(1) - self.data.min(1)))

        # Factors
        A2 = self.control_chart_factors.loc[self.n, 'A2']
        d2 = self.control_chart_factors.loc[self.n, 'd2']
        d3 = self.control_chart_factors.loc[self.n, 'd3']
        D1 = self.control_chart_factors.loc[self.n, 'D1']
        D2 = self.control_chart_factors.loc[self.n, 'D2']
        D3 = self.control_chart_factors.loc[self.n, 'D3']
        D4 = self.control_chart_factors.loc[self.n, 'D4']

        if mu and not sigma or sigma and not mu:
            raise ValueError('Provide both mu and sigma')

        elif mu and sigma:  # mu and sigma known
            if limits:  # Sigma limits given
                # X-bar Control Limits
                cl1 = mu
                ucl1 = mu + limits * (sigma / math.sqrt(self.n))
                lcl1 = mu - limits * (sigma / math.sqrt(self.n))

                # R Control Limits
                cl2 = d2 * sigma
                ucl2 = cl2 + limits * d3 * sigma
                lcl2 = cl2 - limits * d3 * sigma

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('R Chart - Sigma Known - Limits Given',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-bar Chart', 'R Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

            else:  # Sigma limits not given
                # X-bar Control Limits
                cl1 = mu
                ucl1 = mu + stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))
                lcl1 = mu - stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))

                # R Control Limits
                cl2 = d2 * sigma
                ucl2 = D2 * sigma
                lcl2 = D1 * sigma

                # Validate that points are in control
                self.validate_chart_points(f'X-bar Chart - Mu and Sigma Known - Standard Limits - Alpha={alpha}',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points(f'R Chart - Sigma Known - Standard Limits',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

        else:  # mu and sigma unknown
            if limits:  # Sigma limits given
                # X-bar Control Limits
                cl1 = statistics.mean(chart_data['x_bar'])
                ucl1 = cl1 + A2 * statistics.mean(chart_data['R'])
                lcl1 = cl1 - A2 * statistics.mean(chart_data['R'])

                # R Control Limits
                cl2 = statistics.mean(chart_data['R'])
                ucl2 = cl2 * (d2 + limits * d3) / d2
                lcl2 = cl2 * (d2 - limits * d3) / d2

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Unknown - Limits Given',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('R Chart - Sigma Unknown - Limits Given',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

            else:  # Sigma limits not given
                # X-bar Control Limits
                cl1 = statistics.mean(chart_data['x_bar'])
                ucl1 = cl1 + A2 * statistics.mean(chart_data['R'])
                lcl1 = cl1 - A2 * statistics.mean(chart_data['R'])

                # R Control Limits
                cl2 = statistics.mean(chart_data['R'])
                ucl2 = cl2 * D4
                lcl2 = cl2 * D3

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Unknown - Standard Limits',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('R Chart - Sigma Unknown - Standard Limits',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

    # X-bar (Stddev) Chart ----------------------------------------------------
    def xbars_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        """
        The x-bar "stdev" chart is used to monitor the mean and variation of a
        process based on samples taken in a given time. On the x-bar chart, the
        y-axis shows the grand mean and the control limits while the x-axis
        shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: n > 10

        Data Input:
            Greater than 10 vectors or greater than 10 dataframe columns
            (one row = one sample)
        """
        # n must must be greater than 10 (n>10)
        if self.n > 10:
            pass
        else:
            raise ValueError(
                'X-bar(Stddev) chart must have sample size(n) = n > 10')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculated stddev
        chart_data = chart_data.assign(S=self.data.std(1))

        # Factors
        A3 = self.control_chart_factors.loc[self.n, 'A3']
        c4 = self.control_chart_factors.loc[self.n, 'c4']

        if mu and not sigma or sigma and not mu:
            raise ValueError('Provide both mu and sigma')

        elif mu and sigma:  # Mu and Sigma known
            if limits:  # Sigma limits given
                # Control Limits
                cl = mu
                ucl = mu + limits * (sigma / math.sqrt(self.n))
                lcl = mu - limits * (sigma / math.sqrt(self.n))

                # Validate that points are in control
                self.validate_chart_points('X-bar (Stddev) Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = mu
                ucl = mu + stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))
                lcl = mu - stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))

                # Validate that points are in control
                self.validate_chart_points('X-bar (Stddev) Chart - Mu and Sigma Known - Standard Limits',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

        else:  # Mu and Sigma unknown
            if limits:  # Sigma limits given
                # Control Limits
                cl = statistics.mean(chart_data['x_bar'])
                ucl = cl + limits * \
                    (statistics.mean(chart_data['S']) / c4) / math.sqrt(self.n)
                lcl = cl - limits * \
                    (statistics.mean(chart_data['S']) / c4) / math.sqrt(self.n)

                # Validate that points are in control
                self.validate_chart_points('X-bar (Stddev) Chart - Mu and Sigma Unknown - Limits Given',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = statistics.mean(chart_data['x_bar'])
                ucl = cl + A3 * statistics.mean(chart_data['S'])
                lcl = cl - A3 * statistics.mean(chart_data['S'])

                # Validate that points are in control
                self.validate_chart_points('X-bar (Stddev) Chart - Mu and Sigma Unknown - Standard Limits',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #',
                                 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

    # S Chart -----------------------------------------------------------------
    def s_chart(self, sigma=None, limits=None):
        """
        The s-chart is used to monitor the mean and variation of a
        process based on samples taken in a given time. On the s chart, the
        y-axis shows the standard deviation grand mean and the control limits
        while the x-axis shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: n > 10

        Data Input:
            Greater than 10 vectors or greater than 10 dataframe columns
            (one row = one sample)
        """
        # n must must be greater than 10 (n>10)
        if self.n > 10:
            pass
        else:
            raise ValueError('S chart must have sample size(n) = n > 10')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculated stddev
        chart_data = chart_data.assign(S=self.data.std(1))

        # Factors
        c4 = self.control_chart_factors.loc[self.n, 'c4']
        B3 = self.control_chart_factors.loc[self.n, 'B3']
        B4 = self.control_chart_factors.loc[self.n, 'B4']
        B5 = self.control_chart_factors.loc[self.n, 'B5']
        B6 = self.control_chart_factors.loc[self.n, 'B6']

        if sigma:  # Sigma known
            if limits:  # Sigma limits given
                # Control Limits
                cl = c4 * sigma
                ucl = cl + limits * sigma * math.sqrt(1 - c4 ** 2)
                lcl = cl - limits * sigma * math.sqrt(1 - c4 ** 2)

                # Validate that points are in control
                self.validate_chart_points('S Chart - Sigma Known - Limits Given',
                                           chart_data['S'], ucl, lcl)

                # Plot the data
                self.single_plot('S Chart', 'Sample #',
                                 'S Values', chart_data['S'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = c4 * sigma
                ucl = B6 * sigma
                lcl = B5 * sigma

                # Validate that points are in control
                self.validate_chart_points('S Chart - Sigma Known - Standard Limits',
                                           chart_data['S'], ucl, lcl)

                # Plot the data
                self.single_plot('S Chart', 'Sample #',
                                 'S Values', chart_data['S'], ucl, cl, lcl)

        else:  # Sigma unknown
            if limits:  # Sigma limits given
                # Control Limits
                cl = statistics.mean(chart_data['S'])
                ucl = cl * (1 + limits * (math.sqrt(1 - c4 ** 2) / c4))
                lcl = cl * (1 - limits * (math.sqrt(1 - c4 ** 2) / c4))

                # Validate that points are in control
                self.validate_chart_points('S Chart - Sigma Unknown - Limits Given',
                                           chart_data['S'], ucl, lcl)

                # Plot the data
                self.single_plot('S Chart', 'Sample #',
                                 'S Values', chart_data['S'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = statistics.mean(chart_data['S'])
                ucl = cl * B4
                lcl = cl * B3

                # Validate that points are in control
                self.validate_chart_points('S Chart - Sigma Unknown - Standard Limits',
                                           chart_data['S'], ucl, lcl)

                # Plot the data
                self.single_plot('S Chart', 'Sample #',
                                 'S Values', chart_data['S'], ucl, cl, lcl)

    # X-bar and S Chart -------------------------------------------------------
    def xbar_s_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        """
        The x-bar and s chart are used to monitor the mean and variation of a
        process based on samples taken in a given time. In order to use the s
        chart along with the x-bar chart, the sample size n must be greater
        than 10 units. On the x-bar chart, the y-axis shows the grand mean and
        the control limits while the x-axis shows the sample group. On the s
        chart, the y-axis shows the standard deviation grand mean and the
        control limits while the x-axis shows the sample group.

        Criteria:
            Data type: Continuous
            Sample Size: n > 10

        Data Input:
            Greater than 10 vectors or greater than 10 dataframe columns
            (one row = one sample)
        """
        # n must must be greater than 10 (n>10)
        if self.n > 10:
            pass
        else:
            raise ValueError(
                'X-bar & S charts must have sample size(n) = n > 10')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculated stddev
        chart_data = chart_data.assign(S=self.data.std(1))

        # Factors
        A3 = self.control_chart_factors.loc[self.n, 'A3']
        c4 = self.control_chart_factors.loc[self.n, 'c4']
        B3 = self.control_chart_factors.loc[self.n, 'B3']
        B4 = self.control_chart_factors.loc[self.n, 'B4']
        B5 = self.control_chart_factors.loc[self.n, 'B5']
        B6 = self.control_chart_factors.loc[self.n, 'B6']

        if mu and not sigma or sigma and not mu:
            raise ValueError('Provide both mu and sigma')

        elif mu and sigma:  # mu and sigma known
            if limits:  # Sigma limits given
                # X-bar Control Limits
                cl1 = mu
                ucl1 = mu + limits * (sigma / math.sqrt(self.n))
                lcl1 = mu - limits * (sigma / math.sqrt(self.n))

                # S Control Limits
                cl2 = c4 * sigma
                ucl2 = cl2 + limits * sigma * math.sqrt(1 - c4 ** 2)
                lcl2 = cl2 - limits * sigma * math.sqrt(1 - c4 ** 2)

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('S Chart - Sigma Known - Limits Given',
                                           chart_data['S'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-bar Chart', 'S Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'S Values'),
                                 (chart_data['x_bar'], chart_data['S']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

            else:  # Sigma limits not given
                # X-bar Control Limits
                cl1 = mu
                ucl1 = mu + stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))
                lcl1 = mu - stats.norm.ppf(1 - alpha / 2) * \
                    (sigma / math.sqrt(self.n))

                # S Control Limits
                cl2 = c4 * sigma
                ucl2 = B6 * sigma
                lcl2 = B5 * sigma

                # Validate that points are in control
                self.validate_chart_points(f'X-bar Chart - Mu and Sigma Known - Standard Limits - Alpha={alpha}',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('S Chart - Sigma Known - Standard Limits',
                                           chart_data['S'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'S Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'S Values'),
                                 (chart_data['x_bar'], chart_data['S']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

        else:  # mu and sigma unknown
            if limits:  # Sigma limits given
                # X-bar Control Limits
                cl1 = statistics.mean(chart_data['x_bar'])
                ucl1 = cl1 + limits * \
                    (statistics.mean(chart_data['S']) / c4) / math.sqrt(self.n)
                lcl1 = cl1 - limits * \
                    (statistics.mean(chart_data['S']) / c4) / math.sqrt(self.n)

                # S Control Limits
                cl2 = statistics.mean(chart_data['S'])
                ucl2 = cl2 * (1 + limits * (math.sqrt(1 - c4 ** 2) / c4))
                lcl2 = cl2 * (1 - limits * (math.sqrt(1 - c4 ** 2) / c4))

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Unknown - Limits Given',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('S Chart - Sigma Unknown - Limits Given',
                                           chart_data['S'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'S Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'S Values'),
                                 (chart_data['x_bar'], chart_data['S']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

            else:  # Sigma limits not given
                # X-bar Control Limits
                cl1 = statistics.mean(chart_data['x_bar'])
                ucl1 = cl1 + A3 * statistics.mean(chart_data['S'])
                lcl1 = cl1 - A3 * statistics.mean(chart_data['S'])

                # S Control Limits
                cl2 = statistics.mean(chart_data['S'])
                ucl2 = cl2 * B4
                lcl2 = cl2 * B3

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Unknown - Standard Limits',
                                           chart_data['x_bar'], ucl1, lcl1)
                print()
                self.validate_chart_points('S Chart - Sigma Unknown - Standard Limits',
                                           chart_data['S'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'S Chart'),
                                 (None, 'Sample #'),
                                 ('X-bar Values', 'S Values'),
                                 (chart_data['x_bar'], chart_data['S']),
                                 (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

    # c Chart -----------------------------------------------------------------
    def c_chart(self, c=None, limits=None):
        """
        The c-chart is used to monitor the total count of defects in fixed
        samples of size n. The y-axis shows the number of nonconformities per
        sample while the x-axis shows the sample group.

        Criteria:
            Data type: Discrete
            Defects: Multiple per unit (Several defects before defective)
            Constant sample size: Yes

        Data Input:
            A single vector or a single dataframe column of nonconforming count
        """

        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            message = 'c-charts should only consist of one vector'\
                      ' containing the number of defects'
            raise ValueError(message)

        # Data for plot
        chart_data = self.data

        # Calculate c value
        if not c:
            c = statistics.mean(chart_data['x1'])

        # Control Limits
        cl = c
        ucl = c + limits * math.sqrt(c) if limits else c + 3 * math.sqrt(c)
        lcl = c - limits * math.sqrt(c) if limits else c - 3 * math.sqrt(c)

        if lcl < 0:
            lcl = 0

        # Validate that points are in control
        self.validate_chart_points('c Chart', chart_data['x1'], ucl, lcl)

        # Plot the data
        self.single_plot('c Chart', 'Sample #', '# Defects',
                         chart_data['x1'], ucl, cl, lcl)

    # u Chart -----------------------------------------------------------------
    def u_chart(self, u=None, limits=None):
        """
        The u-chart is used to monitor the total count of defects per unit in
        different samples of size n; it assumes that units can have more than a
        single defect. The y-axis shows the number of defects per single unit
        while the x-axis shows the sample group.

        Criteria:
            Data type: Discrete
            Defects: Multiple per unit (Several defects before defective)
            Constant sample size: No

        Data Input:
            First a single vector or a single dataframe column of nonconforming
            and
            Second a single vector or a single dataframe column of sample size
        """

        # n must be equal to 2
        if self.n == 2:
            pass
        else:
            message = 'u-charts should only consist of two vectors'\
                      ' first containing the number of defects'\
                      ' second containing the sample size'
            raise ValueError(message)

        # Data for plot
        chart_data = self.data
        chart_data['u'] = chart_data['x1'] / chart_data['x2']

        # Calculate c value
        if not u:
            u = statistics.mean(chart_data['u'])

        # Control Limits
        cl = u

        if limits:
            ucl = u + limits * np.sqrt(u / chart_data['x2'])
            lcl = u - limits * np.sqrt(u / chart_data['x2'])
        else:
            ucl = u + 3 * np.sqrt(u / chart_data['x2'])
            lcl = u - 3 * np.sqrt(u / chart_data['x2'])

        # Validate that points are in control
        self.validate_chart_points_unequal_samples('u Chart', chart_data['u'],
                                                   ucl, lcl)

        # Plot the data
        self.single_plot_unequal_samples('u Chart', 'Sample #', 'Proportion Defects',
                                         chart_data['u'], ucl, cl, lcl)

    # np Chart ----------------------------------------------------------------

    def np_chart(self, n, p=None, limits=None):
        """
        The np-chart is used to monitor the count of nonconforming units in
        fixed samples of size n. The y-axis shows the total count of
        nonconforming units while the x-axis shows the sample group.

        Criteria:
            Data type: Discrete
            Defects: One per unit
            Constant sample size: Yes

        Data Input:
            A single vector or a single dataframe column of nonconforming count
        """

        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            message = 'np-charts should only consist of one vector'\
                      ' containing the number of defectives'
            raise ValueError(message)

        # Data for plot
        chart_data = self.data
        chart_data['p'] = chart_data['x1'] / n

        # Calculate c value
        if not p:
            p = statistics.mean(chart_data['p'])

        # Control Limits
        cl = n * p

        if limits:
            ucl = n * p + limits * math.sqrt(n * p * (1 - p))
            lcl = n * p - limits * math.sqrt(n * p * (1 - p))
        else:
            ucl = n * p + 3 * math.sqrt(n * p * (1 - p))
            lcl = n * p - 3 * math.sqrt(n * p * (1 - p))

        if lcl < 0:
            lcl = 0

        # Validate that points are in control
        self.validate_chart_points('np Chart', chart_data['x1'], ucl, lcl)

        # Plot the data
        self.single_plot('np Chart', 'Sample #', '# Defective',
                         chart_data['x1'], ucl, cl, lcl)

    # p Chart -----------------------------------------------------------------
    def p_chart(self, p=None, limits=None):
        """
        The p-chart is used to monitor the proportion of nonconforming units in
        different samples of size n; it is based on the binomial distribution
        where each unit has only two possibilities (i.e. defective or not
        defective). The y-axis shows the proportion of nonconforming units
        while the x-axis shows the sample group.

        Criteria:
            Data type: Discrete
            Defects: One per unit
            Constant sample size: No

        Data Input:
            A single vector or a single dataframe column of nonconforming count
            and
            A single vector or a single dataframe column of sample size
        """

        # n must be equal to 2
        if self.n == 2:
            pass
        else:
            message = 'u-charts should only consist of two vectors'\
                      ' first containing the number of defective'\
                      ' second containing the sample size'
            raise ValueError(message)

        # Data for plot
        chart_data = self.data
        chart_data['p'] = chart_data['x1'] / chart_data['x2']

        # Calculate c value
        if not p:
            p = statistics.mean(chart_data['p'])

        # Control Limits
        cl = p

        if limits:
            ucl = p + limits * np.sqrt((p * (1 - p)) / chart_data['x2'])
            lcl = p - limits * np.sqrt((p * (1 - p)) / chart_data['x2'])
        else:
            ucl = p + 3 * np.sqrt((p * (1 - p)) / chart_data['x2'])
            lcl = p - 3 * np.sqrt((p * (1 - p)) / chart_data['x2'])

        # Validate that points are in control
        self.validate_chart_points_unequal_samples('p Chart', chart_data['p'],
                                                   ucl, lcl)

        # Plot the data
        self.single_plot_unequal_samples('p Chart', 'Sample #', 'Proportion Defective',
                                         chart_data['p'], ucl, cl, lcl)


np.random.seed(42)

df = pd.DataFrame(np.random.randint(1, 5, 10).tolist())
df['a'] = np.random.randint(20, 40, size=(10, 1))

cc = ControlChart(data=df)
cc.p_chart()
