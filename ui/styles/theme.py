def apply_theme(app):
    app.setStyleSheet("""
        QWidget {
            font-family: Segoe UI;
            background: #f5f7fb;
        }

        QPushButton {
            background: #e5e7eb;
            border-radius: 8px;
            padding: 10px;
        }

        QPushButton:hover {
            background: #d1d5db;
        }
    """)