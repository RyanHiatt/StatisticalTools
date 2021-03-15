import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics


class ControlChart:

    def __init__(self, data):

        self.data = self.standardize_data(data)
        self.n = len(self.data.columns)
        self.datatype = self.check_datatype(self.data)

        self.control_chart_factors = pd.read_csv('control_chart_factors.csv', index_col='n')

        # Plot Variables



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
        # formatted_data.index = range(1, len(formatted_data.values) + 1)
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
        print(self.data.describe())

    @staticmethod
    def chart_summary(points, chart_factors, cl, ucl, lcl):
        print(f'Data:\n{points}\n')
        print('Chart Factors:')
        for factor in chart_factors:
            print(f'\t{factor}= {chart_factors[factor]}')
        print(f'Upper Control Limit = {ucl}')
        print(f'Center Line = {cl}')
        print(f'Lower Control Limit = {lcl}')
        print()

    @staticmethod
    def validate_chart_points(chart_name, point, ucl, lcl):
        points_out_of_control = False
        for i in range(len(point)):
            if point[i] > ucl or point[i] < lcl:
                points_out_of_control = True
            else:
                continue
        if points_out_of_control:
            print(chart_name, '- Control Chart Points: Out of Control')
            for i in range(len(point)):
                if point[i] > ucl or point[i] < lcl:
                    print(f'Point {i} is out of control with a value of {point.loc[i]}')
                else:
                    continue
        else:
            print(chart_name, '- Control Chart Points: In Control')
        print()

    def auto_plot_chart(self):
        if self.datatype == 'continuous':
            print('continuous plot coming up')

        elif self.datatype == 'discrete':
            print('discrete plot coming up')

        else:
            raise ValueError('Invalid Data, data must be a pandas dataframe with only raw data.')

    # X Chart ----------------------------------------------------------------------------------------------------------
    def x_chart(self):
        mr = pd.Series([np.nan], dtype='object')
        for i in range(1, len(self.data)):
            mr.loc[i] = (abs(self.data.iloc[i].item() - self.data.iloc[i - 1].item()))

        # Plot Data
        chart_data = self.data.assign(mR=mr.values)

        # Plot Factors
        d2 = self.control_chart_factors.loc[2, 'd2']

        # Plot Control Limits
        cl = statistics.mean(chart_data['x1'])
        ucl = cl + 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2
        lcl = cl - 3 * statistics.mean(chart_data['mR'][1:len(chart_data['mR'])]) / d2

        self.chart_summary(points=chart_data['x1'], chart_factors={'d2': d2}, cl=cl, ucl=ucl, lcl=lcl)

        # Create X Chart
        fig, axs = plt.subplots()
        axs.plot(chart_data['x1'], linestyle='-', marker='o', color='slategray')  # Plot Data
        axs.axhline(cl, color='blue')  # CL
        axs.axhline(ucl, linestyle='--', color='red')  # UCL
        axs.axhline(lcl, linestyle='--', color='red')  # LCL
        axs.set(title='X Control Chart', xlabel='Sample #', ylabel='Data Values')  # Titles and Labels

        self.validate_chart_points('X', chart_data['x1'], ucl, lcl)

        plt.show()

    def mR_chart(self):
        pass

    def x_mR_chart(self):
        pass

    def range_xbar_chart(self):
        pass

    def r_chart(self):
        pass

    def xbar_range_chart(self):
        pass

    def stddev_xbar_chart(self):
        pass

    def s_chart(self):
        pass

    def xbar_s_chart(self):
        pass

    def c_chart(self):
        pass

    def u_chart(self):
        pass

    def np_chart(self):
        pass

    def p_chart(self):
        pass


# df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list('ABCD'))
df = pd.DataFrame(np.random.rand(50, 1) * 100, columns=list('A'))
# df = [[3, 5, 2, 3, 5], [5, 9, 2, 6, 4], [1, 5, 3, 4, 2], [7, 8, 6, 0, 1]]
# df = np.random.randint(0, 100, size=(50, 6))

cc = ControlChart(data=df)

print(cc.x_chart())
