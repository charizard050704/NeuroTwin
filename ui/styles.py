def apply_styles(window):
    window.setStyleSheet("""
        QWidget {
            background-color: #121212;
            color: white;
        }
        QPushButton {
            padding: 8px;
            background: #1f1f1f;
            border-radius: 6px;
        }
    """)