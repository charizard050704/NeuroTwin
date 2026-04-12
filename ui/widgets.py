from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

    def clear(self):
        self.ax.clear()

    def plot_prediction(self, actual, pred):
        self.clear()
        self.ax.plot(actual, label="Actual")
        self.ax.plot(pred, label="Predicted")
        self.ax.legend()
        self.draw()

    def plot_anomalies(self, actual, pred, idx):
        self.clear()
        self.ax.plot(actual, label="Actual")
        self.ax.plot(pred, label="Predicted")
        self.ax.scatter(idx, pred[idx], color='red')
        self.ax.legend()
        self.draw()

    def plot_forecast(self, future):
        self.clear()
        self.ax.plot(future, label="Future")
        self.ax.legend()
        self.draw()

    def plot_whatif(self, orig, mod):
        self.clear()
        self.ax.plot(orig, label="Original")
        self.ax.plot(mod, label="Modified")
        self.ax.legend()
        self.draw()