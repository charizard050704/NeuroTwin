from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from ui.widgets import PlotCanvas
from ui.backend_adapter import NeuroTwinBackend
from ui.styles import apply_styles


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NeuroTwin")
        self.setGeometry(100, 100, 1200, 700)

        apply_styles(self)

        self.backend = NeuroTwinBackend()

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        self.status = QLabel("Ready")

        self.plot = PlotCanvas(self)

        self.btn_pred = QPushButton("Prediction")
        self.btn_anom = QPushButton("Anomalies")
        self.btn_fore = QPushButton("Forecast")
        self.btn_what = QPushButton("What-If")

        layout.addWidget(self.status)
        layout.addWidget(self.btn_pred)
        layout.addWidget(self.btn_anom)
        layout.addWidget(self.btn_fore)
        layout.addWidget(self.btn_what)
        layout.addWidget(self.plot)

        central.setLayout(layout)

        self.btn_pred.clicked.connect(self.show_prediction)
        self.btn_anom.clicked.connect(self.show_anomaly)
        self.btn_fore.clicked.connect(self.show_forecast)
        self.btn_what.clicked.connect(self.show_whatif)

    def show_prediction(self):
        actual, pred = self.backend.run_prediction()
        self.plot.plot_prediction(actual, pred)
        self.status.setText("Prediction done")

    def show_anomaly(self):
        actual, pred, idx = self.backend.run_anomaly()
        self.plot.plot_anomalies(actual, pred, idx)
        self.status.setText("Anomalies detected")

    def show_forecast(self):
        future = self.backend.run_forecast()
        self.plot.plot_forecast(future)
        self.status.setText("Forecast done")

    def show_whatif(self):
        orig, mod = self.backend.run_whatif()
        self.plot.plot_whatif(orig, mod)
        self.status.setText("What-if done")