from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel


class Topbar(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(55)

        # 🔥 NO WHITE BOX
        self.setStyleSheet("""
            QWidget {
                background: transparent;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("NeuroTwin Dashboard")
        title.setStyleSheet("""
            font-size:16px;
            font-weight:600;
            color:white;
        """)

        self.status = QLabel("● Ready")
        self.status.setStyleSheet("""
            color:#22c55e;
            font-size:13px;
        """)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status)