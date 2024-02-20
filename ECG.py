import pickle

import numpy as np
import neurokit2 as nk  # Load the package
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

sample_rate = 500

lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ECG:
    def __init__(self):
        self.heart_rate = np.random.randint(50, 120)
        ecg = nk.ecg_simulate(duration=8, sampling_rate=sample_rate, noise=0.1, heart_rate=self.heart_rate,
                              method="multileads")
        self.signals = np.transpose(ecg.to_numpy())
        self.names = list(ecg.columns)

    def get_signals(self):
        return self.signals

    def get_name(self, index):
        if index < len(self.names):
            return self.names[index]
        else:
            raise IndexError("Index out of range")


    def plot(self,
            sample_rate=500,
            title='ECG 12',
            lead_index=lead_index,
            lead_order=None,
            style=None,
            columns=[2],
            row_height=6,
            show_lead_name=True,
            show_grid=True,
            show_separate_line=True,
    ):
        """Plot multi lead ECG chart.
        # Arguments
            ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
            sample_rate: Sample rate of the signal.
            title      : Title which will be shown on top off chart
            lead_index : Lead name array in the same order of ecg, will be shown on
                left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            lead_order : Lead display order
            columns    : display columns, defaults to 2
            style      : display style, defaults to None, can be 'bw' which means black white
            row_height :   how many grid should a lead signal have,
            show_lead_name : show lead name
            show_grid      : show grid
            show_separate_line  : show separate line
        """
        ecg = self.get_signals()
        assert len(ecg) >= np.sum(columns)
        if not lead_order:
            lead_order = list(range(0, len(ecg)))
        secs = len(ecg[0]) / sample_rate * columns[0]
        rows = len(columns)
        display_factor = 1
        line_width = 0.5
        fig, ax = plt.subplots(figsize=(secs * display_factor, rows * row_height / 5 * display_factor))
        display_factor = display_factor ** 0.5
        fig.subplots_adjust(
            hspace=0,
            wspace=0,
            left=0,  # the left side of the subplots of the figure
            right=1,  # the right side of the subplots of the figure
            bottom=0,  # the bottom of the subplots of the figure
            top=1
        )

        fig.suptitle(title)

        x_min = 0
        x_max = secs
        y_min = row_height / 4 - (rows / 2) * row_height
        y_max = row_height / 4

        if (style == 'bw'):
            color_major = (0.4, 0.4, 0.4)
            color_minor = (0.75, 0.75, 0.75)
            color_line = (0, 0, 0)
        else:
            color_major = (1, 0, 0)
            color_minor = (1, 0.7, 0.7)
            color_line = (0, 0, 0.7)

        if (show_grid):
            ax.set_xticks(np.arange(x_min, x_max, 0.2))
            ax.set_yticks(np.arange(y_min, y_max, 0.5))
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
            ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)

        for r, c in enumerate(columns):
            for i in range(0, c):
                y_offset = -(row_height / 2) * r
                x_offset = secs / c * i
                t_lead = lead_order[c * r + i]
                if (show_separate_line):
                    ax.plot([x_offset, x_offset],
                            [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3],
                            linewidth=line_width * display_factor, color=color_line)
                step = secs / len(ecg[t_lead]) / c
                if (show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line
                )

    def show(self):
        plt.show()

    def save_as_png(self,file_name, dpi=100):
        """Plot multi lead ECG chart.
        # Arguments
            file_name: file_name
            path     : path to save image, defaults to current folder
            dpi      : set dots per inch (dpi) for the saved image
            layout   : Set equal to "tight" to include ax labels on saved image
        """
        plt.savefig('./' + file_name + '.png', dpi=dpi, bbox_inches=None)
        plt.close()

    def save_label(self, type ,filename):
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        label = f"{type}"
        for x, y in corners:
            label += f" {x:.6f} {y:.6f}"
        label += "\n"
        with open(filename, 'w') as f:
            f.write(label)

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
