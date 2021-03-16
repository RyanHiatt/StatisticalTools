import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
from scipy import stats


"""
Control Chart Package
Version 0.1.0
Created by: Ryan Hiatt
Last Updated: 03/16/2021
"""


class ControlChart:

    def __init__(self, data):

        self.data = self.standardize_data(data)
        self.n = len(self.data.columns)
        self.datatype = self.check_datatype(self.data)

        # Control Chart Variables
        self.control_chart_factors = pd.read_csv('control_chart_factors.csv', index_col='n')

    """
    Chart types:
        Continuous
            X Chart
            mR Chart
            X and mR Charts
            
            range X-bar Chart
            R Chart
            X-bar and R Charts
            
            stddev X-bar Chart
            S Chart
            X-bar and S Chart
        Discrete
            C Chart
            U Chart
            np Chart
            p Chart
    """

    @staticmethod
    def standardize_data(data):
        formatted_data = pd.DataFrame(data)
        formatted_data.columns = [f'x{y+1}' for y in range(len(formatted_data.columns))]
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
                print(f'Point {i} is out of control with a value of {points.loc[i]}')
        if control:
            print('All points in control.')

    @staticmethod
    def single_plot(title, xlab, ylab, data, ucl, cl, lcl):
        # Create Single Chart
        fig, ax = plt.subplots(1, 1, sharex=True, squeeze=True, figsize=(10, 3.5), tight_layout=True)

        ax.plot(data, linestyle='-', marker='o', color='slategray')  # Plot Data
        ax.axhline(ucl, linestyle='--', color='red', label=f'UCL = {ucl:.2f}')  # UCL
        ax.axhline(cl, color='blue', label=f'CL = {cl:.2f}')  # CL
        ax.axhline(lcl, linestyle='--', color='red', label=f'LCL = {lcl:.2f}')  # LCL
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)  # Plot Legend
        ax.set(title=title, xlabel=xlab, ylabel=ylab)  # Titles and Labels

        return plt.show()

    @staticmethod
    def double_plot(title, xlab, ylab, data, ucl, cl, lcl):
        # Create Double Chart
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(10, 7), tight_layout=True)

        # Plot One
        ax1.plot(data[0], linestyle='-', marker='o', color='slategray')  # Plot Data
        ax1.axhline(ucl[0], linestyle='--', color='red', label=f'UCL = {ucl[0]:.2f}')  # UCL
        ax1.axhline(cl[0], color='blue', label=f'CL = {cl[0]:.2f}')  # CL
        ax1.axhline(lcl[0], linestyle='--', color='red', label=f'LCL = {lcl[0]:.2f}')  # LCL
        ax1.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)  # Plot Legend
        ax1.set(title=title[0], xlabel=xlab[0], ylabel=ylab[0])  # Titles and Labels

        # Plot Two
        ax2.plot(data[1], linestyle='-', marker='o', color='slategray')  # Plot Data
        ax2.axhline(ucl[1], linestyle='--', color='red', label=f'UCL = {ucl[1]:.2f}')  # UCL
        ax2.axhline(cl[1], color='blue', label=f'CL = {cl[1]:.2f}')  # CL
        ax2.axhline(lcl[1], linestyle='--', color='red', label=f'LCL = {lcl[1]:.2f}')  # LCL
        ax2.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
        ax2.set(title=title[1], xlabel=xlab[1], ylabel=ylab[1])  # Titles and Labels

        return plt.show()

    # def auto_plot_chart(self):
    #     if self.datatype == 'continuous':
    #         print('continuous plot coming up')
    #
    #     elif self.datatype == 'discrete':
    #         print('discrete plot coming up')
    #
    #     else:
    #         raise ValueError('Invalid Data, data must be a pandas dataframe with only raw data.')

    # X Chart ----------------------------------------------------------------------------------------------------------
    def x_chart(self):
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('X chart must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item() - self.data.iloc[i - 1].item()))

        # Data for Plot
        chart_data = self.data.assign(mR=mr.values)

        # Factors
        d2 = self.control_chart_factors.loc[2, 'd2']

        # Control Limits
        cl = statistics.mean(chart_data['x1'])
        ucl = cl + 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2
        lcl = cl - 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2

        # Validate that points are in control
        self.validate_chart_points('X Chart', chart_data['x1'], ucl, lcl)

        # Plot the data
        self.single_plot('X Chart', 'Sample #', 'Data Values', chart_data['x1'], ucl, cl, lcl)

    # mR Chart ---------------------------------------------------------------------------------------------------------
    def mR_chart(self):
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('mR chart must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item() - self.data.iloc[i - 1].item()))

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
        self.single_plot('mR Chart', 'Sample #', 'mR Values', chart_data['mR'], ucl, cl, lcl)

        plt.show()

    # X & mR Chart -----------------------------------------------------------------------------------------------------
    def x_mR_chart(self):
        # n must be equal to 1
        if self.n == 1:
            pass
        else:
            raise ValueError('X & mR charts must have sample size(n) = 1')

        # Calculate Moving Range
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item() - self.data.iloc[i - 1].item()))

        # Data for Plot
        chart_data = self.data.assign(mR=mr.values)

        # Factors
        d2 = self.control_chart_factors.loc[self.n + 1, 'd2']
        D3 = self.control_chart_factors.loc[self.n + 1, 'D3']
        D4 = self.control_chart_factors.loc[self.n + 1, 'D4']

        # X Control Limits
        cl1 = statistics.mean(chart_data['x1'])
        ucl1 = cl1 + 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2
        lcl1 = cl1 - 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2

        # mR Control Limits
        cl2 = statistics.mean(chart_data['mR'][1:len(chart_data['mR'])])
        ucl2 = cl2 * D4
        lcl2 = cl2 * D3

        # Validate that points are in control
        self.validate_chart_points('X Chart', chart_data['x1'], ucl1, lcl1)
        print()
        self.validate_chart_points('mR Chart', chart_data['mR'], ucl2, lcl2)

        # Plot the data
        self.double_plot(('X Chart', 'mR Chart'), (None, 'Sample #'), ('Data Values', 'mR Values'),
                         (chart_data['x1'], chart_data['mR']), (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

    # X-bar (Range) Chart ----------------------------------------------------------------------------------------------
    def xbarr_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError('X-bar(Range) chart must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(R=abs(self.data.max(1) - self.data.min(1)))

        # Factors
        A2 = self.control_chart_factors.loc[self.n, 'A2']

        if mu and not sigma or sigma and not mu:
            raise ValueError('Provide both mu and sigma')

        elif mu and sigma:  # Mu and Sigma known
            if limits:  # Sigma limits given
                # Control Limits
                cl = mu
                ucl = mu + limits * (self.data.values.std(ddof=1) / math.sqrt(self.n))
                lcl = mu - limits * (self.data.values.std(ddof=1) / math.sqrt(self.n))

                # Validate that points are in control
                self.validate_chart_points('X-bar (Range) Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #', 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = mu
                ucl = mu + stats.norm.ppf(1 - alpha / 2) * (self.data.values.std(ddof=1) / math.sqrt(self.n))
                lcl = mu - stats.norm.ppf(1 - alpha / 2) * (self.data.values.std(ddof=1) / math.sqrt(self.n))

                # Validate that points are in control
                self.validate_chart_points('X-bar (Range) Chart - Mu and Sigma Known - Standard Limits',
                                           chart_data['x_bar'], ucl, lcl)

                # Plot the data
                self.single_plot('X-bar Chart', 'Sample #', 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

        else:  # Mu and Sigma unknown
            # Control Limits
            cl = statistics.mean(chart_data['x_bar'])
            ucl = cl + A2 * statistics.mean(chart_data['R'])
            lcl = cl - A2 * statistics.mean(chart_data['R'])

            # Validate that points are in control
            self.validate_chart_points('X-bar (Range) Chart - Mu and Sigma Unknown - Standard Limits',
                                       chart_data['x_bar'], ucl, lcl)

            # Plot the data
            self.single_plot('X-bar Chart', 'Sample #', 'X-bar Values', chart_data['x_bar'], ucl, cl, lcl)

    # R Chart ----------------------------------------------------------------------------------------------------------
    def R_chart(self, sigma=None, limits=None):
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError('R chart must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(R=abs(self.data.max(1) - self.data.min(1)))

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
                self.single_plot('R Chart', 'Sample #', 'R Values', chart_data['R'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = d2 * sigma
                ucl = D2 * sigma
                lcl = D1 * sigma

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Known - Standard Limits',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #', 'R Values', chart_data['R'], ucl, cl, lcl)

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
                self.single_plot('R Chart', 'Sample #', 'R Values', chart_data['R'], ucl, cl, lcl)

            else:  # Sigma limits not given
                # Control Limits
                cl = statistics.mean(chart_data['R'])
                ucl = cl * D4
                lcl = cl * D3

                # Validate that points are in control
                self.validate_chart_points('R Chart - Sigma Unknown - Standard Limits',
                                           chart_data['R'], ucl, lcl)

                # Plot the data
                self.single_plot('R Chart', 'Sample #', 'R Values', chart_data['R'], ucl, cl, lcl)

    # X-bar & R Chart --------------------------------------------------------------------------------------------------
    def xbar_R_chart(self, mu=None, sigma=None, limits=None, alpha=0.05):
        # n must must be between 1 and 11 (1<n<11)
        if 1 < self.n < 11:
            pass
        else:
            raise ValueError('X-bar & R charts must have sample size(n) = 1 < n < 11')

        # Calculate x-bar
        chart_data = pd.DataFrame(self.data.mean(1), columns=['x_bar'])

        # Calculate range
        chart_data = chart_data.assign(R=abs(self.data.max(1) - self.data.min(1)))

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
                ucl1 = mu + limits * (self.data.values.std(ddof=1) / math.sqrt(self.n))
                lcl1 = mu - limits * (self.data.values.std(ddof=1) / math.sqrt(self.n))

                # R Control Limits
                cl2 = d2 * sigma
                ucl2 = cl2 + limits * d3 * sigma
                lcl2 = cl2 - limits * d3 * sigma

                # Validate that points are in control
                self.validate_chart_points('X-bar Chart - Mu and Sigma Known - Limits Given',
                                           chart_data['x_bar'], ucl1, lcl1)
                self.validate_chart_points('R Chart - Sigma Known - Limits Given',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-bar Chart', 'R Chart'), (None, 'Sample #'), ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']), (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

            else:  # Sigma limits not given
                # X-bar Control Limits
                cl1 = mu
                ucl1 = mu + stats.norm.ppf(1 - alpha / 2) * (self.data.values.std(ddof=1) / math.sqrt(self.n))
                lcl1 = mu - stats.norm.ppf(1 - alpha / 2) * (self.data.values.std(ddof=1) / math.sqrt(self.n))

                # R Control Limits
                cl2 = d2 * sigma
                ucl2 = D2 * sigma
                lcl2 = D1 * sigma

                # Validate that points are in control
                self.validate_chart_points(f'X-bar Chart - Mu and Sigma Known - Standard Limits - Alpha={alpha}',
                                           chart_data['x_bar'], ucl1, lcl1)
                self.validate_chart_points(f'R Chart - Sigma Known - Standard Limits',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'), (None, 'Sample #'), ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']), (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

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
                self.validate_chart_points('R Chart - Sigma Unknown - Limits Given',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'), (None, 'Sample #'), ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']), (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

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
                self.validate_chart_points('X-bar(Range) Chart - Mu and Sigma Unknown - Standard Limits',
                                           chart_data['x_bar'], ucl1, lcl1)
                self.validate_chart_points('R Chart - Sigma Unknown - Standard Limits',
                                           chart_data['R'], ucl2, lcl2)

                # Plot the data
                self.double_plot(('X-Bar Chart', 'R Chart'), (None, 'Sample #'), ('X-bar Values', 'R Values'),
                                 (chart_data['x_bar'], chart_data['R']), (ucl1, ucl2), (cl1, cl2), (lcl1, lcl2))

    # X-bar (Stddev) Chart ---------------------------------------------------------------------------------------------
    def xbars_chart(self):
        pass

    # S Chart ----------------------------------------------------------------------------------------------------------
    def s_chart(self):
        pass

    # X-bar and s Chart ------------------------------------------------------------------------------------------------
    def xbar_s_chart(self):
        pass

    # c Chart ----------------------------------------------------------------------------------------------------------
    def c_chart(self):
        pass

    # u Chart ----------------------------------------------------------------------------------------------------------
    def u_chart(self):
        pass

    # np Chart ---------------------------------------------------------------------------------------------------------
    def np_chart(self):
        pass

    # p Chart ----------------------------------------------------------------------------------------------------------
    def p_chart(self):
        pass


# df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list('ABCD'))
# df = pd.DataFrame(np.random.rand(50, 1) * 100, columns=list('A'))
df = pd.DataFrame(np.random.rand(50, 6) * 100)
# df = [[3, 5, 2, 3, 5], [5, 9, 2, 6, 4], [1, 5, 3, 4, 2], [7, 8, 6, 0, 1]]
# df = np.random.randint(0, 100, size=(50, 6))

cc = ControlChart(data=df)

print(cc.xbar_R_chart())
