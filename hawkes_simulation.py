#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
@File    :   hawkes_simulation.py
@Time    :   2022/2/12 8:21 PM
@Author  :   Jinnan Huang
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from functools import partial


# noinspection SpellCheckingInspection,PyPep8Naming
class HawkesSimulation:
    """
    the simulation of Hawkes processes, Ogata (1981)
    """
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    def __init__(self, lambda0, T, Y, delta):
        """
        Initialization
        :type T:
        :param lambda0: λ0(0)
        """
        self.delta = delta
        self.Y = Y
        self.n = 1
        self.T = T
        self.lambda0 = lambda0
        self.lambda_star = lambda0
        self.lambda_s, self.s, self.generated_timestamp_array = (None,) * 3

    def generate_first_event(self):
        """
        First event
        :return:
        """
        u = np.random.uniform(0, 1, 1)[0]
        self.s = (- 1 / self.lambda_star) * np.log(u)
        if self.s <= self.T:
            self.generated_timestamp_array = np.array([self.s])
            logging.info(f'[event {self.n}]: {self.generated_timestamp_array}')
        else:
            return self.generated_timestamp_array

    @staticmethod
    def intensity_function(lambda_0_t, Y, T_array: np.ndarray, delta, t):
        """
        intensity of Hawkes process
        :type delta:
        :param lambda_0_t:
        :param Y:
        :param t:
        :param T_array:
        :return:
        """
        lambda_t_condi_history = lambda_0_t + np.sum(Y * np.exp(- delta * (t - T_array[T_array <= t])))
        return lambda_t_condi_history

    def update_maximum_intensity(self):
        self.lambda_star = self.intensity_function(lambda_0_t=self.lambda0,
                                                   Y=self.Y,
                                                   delta=self.delta,
                                                   t=self.generated_timestamp_array[-1],
                                                   T_array=self.generated_timestamp_array) + self.Y

    def generate_new_event(self):
        u = np.random.uniform(0, 1, 1)[0]
        self.s = self.s - (1 / self.lambda_star) * np.log(u)
        if self.s >= self.T:
            return 'exit'
        else:
            return 'continue'

    def rejection_test(self):
        d = np.random.uniform(0, 1, 1)[0]
        if d <= self.intensity_function(lambda_0_t=self.lambda0,
                                        Y=self.Y,
                                        delta=self.delta,
                                        t=self.s,
                                        T_array=self.generated_timestamp_array) / self.lambda_star:
            self.generated_timestamp_array = np.append(self.generated_timestamp_array, self.s)
            logging.info(f'[event {self.n}]: {self.generated_timestamp_array}')
            return 'passed'
        else:
            return 'failed'

    def general_routine(self):
        while True:
            self.n += 1
            self.update_maximum_intensity()
            code = self.generate_new_event()
            if code is 'exit':
                return self.generated_timestamp_array
            rejection_test_res = self.rejection_test()
            if rejection_test_res is 'passed':
                continue
            elif rejection_test_res is 'failed':
                # update maxmium intensity
                FLAG = True
                while FLAG:
                    # update maxmium intensity
                    self.lambda_star = self.intensity_function(lambda_0_t=self.lambda0,
                                                               Y=self.Y,
                                                               delta=self.delta,
                                                               t=self.s,
                                                               T_array=self.generated_timestamp_array)
                    code = self.generate_new_event()
                    if code is 'exit':
                        return self.generated_timestamp_array
                    sub_rejection_test_result = self.rejection_test()
                    if sub_rejection_test_result is 'passed':
                        FLAG = False

    def simulate(self):
        self.generate_first_event()
        self.general_routine()

    def plot_intensity(self):
        fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
        t_array = np.linspace(0, 30, 3000000)
        intensity_function_map = np.vectorize(partial(self.intensity_function, self.lambda0, self.Y,
                                                      self.generated_timestamp_array, self.delta))
        lambda_array = intensity_function_map(t_array)
        ax.plot(t_array, lambda_array)
        plt.show()


if __name__ == '__main__':
    hk_sim = HawkesSimulation(lambda0=0.5, T=30, Y=0.15, delta=1)
    hk_sim.simulate()
    hk_sim.plot_intensity()
