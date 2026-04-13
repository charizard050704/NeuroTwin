from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame
)
from ui.components.sidebar import Sidebar
from ui.components.topbar import Topbar
from ui.components.graph_view import GraphView
from ui.controller import Controller

from utils.anomaly import detect_anomalies
from inference.what_if import predict_future, what_if_simulation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NeuroTwin Dashboard")
        self.setGeometry(100, 100, 1500, 850)

        self.controller = Controller()

        # 🔥 SINGLE SOURCE + FREEZE STORAGE
        self.base_data = None
        self.base_seq = None
        self.forecast_output = None
        self.whatif_output = None

        self.init_ui()

    def init_ui(self):

        root = QWidget()
        self.setCentralWidget(root)

        root.setStyleSheet("background:#0f172a;")

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.topbar = Topbar()
        outer.addWidget(self.topbar)

        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)

        self.sidebar = Sidebar()
        self.sidebar.setFixedWidth(190)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(12)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self.mse_card = self.metric_card("MSE", "--")
        self.mae_card = self.metric_card("MAE", "--")

        metrics_row.addWidget(self.mse_card)
        metrics_row.addWidget(self.mae_card)

        graph_container = QFrame()
        graph_container.setStyleSheet("""
            QFrame {
                background: #111827;
                border-radius: 14px;
                padding: 12px;
                border: 1px solid #1f2937;
            }
        """)

        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(16, 16, 16, 16)

        self.graph = GraphView()
        graph_layout.addWidget(self.graph)

        content_layout.addLayout(metrics_row)
        content_layout.addWidget(graph_container)

        body.addWidget(self.sidebar)
        body.addLayout(content_layout)

        outer.addLayout(body)

        # ===== CONNECTIONS =====
        self.sidebar.buttons["prediction"].clicked.connect(self.show_pred)
        self.sidebar.buttons["anomaly"].clicked.connect(self.show_anom)
        self.sidebar.buttons["forecast"].clicked.connect(self.show_fore)
        self.sidebar.buttons["whatif"].clicked.connect(self.show_what)

    def metric_card(self, title, value):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: #111827;
                border-radius: 14px;
                padding: 16px;
                border: 1px solid #1f2937;
            }
        """)

        layout = QVBoxLayout(card)

        label = QLabel(title)
        label.setStyleSheet("color:#9ca3af;font-size:12px;")

        value_label = QLabel(value)
        value_label.setStyleSheet("color:white;font-size:22px;font-weight:600;")

        card.value_label = value_label

        layout.addWidget(label)
        layout.addWidget(value_label)

        return card

    # =========================
    # ACTIONS (FINAL FIXED)
    # =========================

    def show_pred(self):
        self.sidebar.set_active("prediction")

        # 🔥 ONLY PLACE WHERE MODEL RUNS
        a, p = self.controller.prediction()

        self.base_data = (a, p)

        # RESET dependent outputs
        self.forecast_output = None
        self.whatif_output = None

        try:
            self.base_seq = self.controller.X_test[0]
        except:
            self.base_seq = None

        self.graph.plot_prediction(a, p)

        mse = ((a - p) ** 2).mean()
        mae = abs(a - p).mean()

        self.mse_card.value_label.setText(f"{mse:.4f}")
        self.mae_card.value_label.setText(f"{mae:.4f}")

    def show_anom(self):
        self.sidebar.set_active("anomaly")

        if self.base_data is None:
            return

        a, p = self.base_data
        idx = detect_anomalies(p, a)

        self.graph.plot_anomaly(a, p, idx)

    def show_fore(self):
        self.sidebar.set_active("forecast")

        if self.base_seq is None:
            return

        # 🔥 RUN ONLY ONCE
        if self.forecast_output is None:
            self.forecast_output = predict_future(self.controller.model, self.base_seq)

        self.graph.plot_forecast(self.forecast_output)

    def show_what(self):
        self.sidebar.set_active("whatif")

        if self.base_seq is None:
            return

        # 🔥 RUN ONLY ONCE
        if self.whatif_output is None:
            result = what_if_simulation(self.controller.model, self.base_seq)

            if isinstance(result, tuple) and len(result) == 3:
                o, m, _ = result
            else:
                o, m = result

            self.whatif_output = (o, m)

        o, m = self.whatif_output
        self.graph.plot_whatif(o, m)