from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedWidth(200)

        self.setStyleSheet("""
            QWidget {
                background: #0f172a;
                border: none;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ===== TITLE (NO BOX, ONLY TEXT) =====
        app_title = QLabel("NeuroTwin AI")
        app_title.setStyleSheet("""
            background: transparent;
            border: none;
            color: white;
            font-size:16px;
            font-weight:600;
        """)
        layout.addWidget(app_title)

        # ===== SECTION LABEL =====
        section = QLabel("Navigation")
        section.setStyleSheet("""
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size:11px;
            margin-top:10px;
        """)
        layout.addWidget(section)

        # ===== BUTTONS =====
        self.buttons = {
            "prediction": self.btn("Prediction"),
            "anomaly": self.btn("Anomalies"),
            "forecast": self.btn("Forecast"),
            "whatif": self.btn("What-If"),
        }

        for b in self.buttons.values():
            layout.addWidget(b)

        layout.addStretch()

    def btn(self, text):
        b = QPushButton(text)
        b.setCursor(Qt.PointingHandCursor)
        b.setMinimumHeight(42)
        b.setStyleSheet(self.default())
        return b

    def default(self):
        return """
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 10px;
                padding: 10px;
                text-align: left;
                color: #ffffff;
                font-size:13px;
            }
            QPushButton:hover {
                background: #1e293b;
            }
        """

    def active(self):
        return """
            QPushButton {
                background: #2563eb;
                border: none;
                border-radius: 10px;
                padding: 10px;
                text-align: left;
                color: white;
                font-weight:500;
            }
        """

    def set_active(self, key):
        for k, b in self.buttons.items():
            b.setStyleSheet(self.default())

        self.buttons[key].setStyleSheet(self.active())