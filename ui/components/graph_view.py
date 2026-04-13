import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout


class GraphView(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")

        # grid + axes
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.getAxis("left").setPen("black")
        self.plot.getAxis("bottom").setPen("black")

        # LEGEND 🔥
        self.legend = self.plot.addLegend()

        layout.addWidget(self.plot)

    def clear(self):
        self.plot.clear()
        self.legend.clear()

    def plot_prediction(self, actual, pred):
        self.clear()

        self.plot.plot(
            actual,
            pen=pg.mkPen("#2563eb", width=2),
            name="Actual"
        )

        self.plot.plot(
            pred,
            pen=pg.mkPen("#f97316", width=2),
            name="Predicted"
        )

    def plot_anomaly(self, actual, pred, idx):
        self.clear()

        self.plot.plot(actual, pen=pg.mkPen("#2563eb", width=2), name="Actual")
        self.plot.plot(pred, pen=pg.mkPen("#f97316", width=2), name="Predicted")

        self.plot.plot(
            idx,
            pred[idx],
            pen=None,
            symbol='o',
            symbolBrush='#ef4444',
            symbolSize=8,
            name="Anomalies"
        )

    def plot_forecast(self, future):
        self.clear()

        self.plot.plot(
            future,
            pen=pg.mkPen("#16a34a", width=3),
            name="Forecast"
        )

    def plot_whatif(self, orig, mod):
        self.clear()

        self.plot.plot(orig, pen=pg.mkPen("#2563eb", width=2), name="Original")
        self.plot.plot(mod, pen=pg.mkPen("#f97316", width=2), name="Modified")