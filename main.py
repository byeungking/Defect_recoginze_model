import sys
from PyQt5.QtWidgets import QApplication
from gui import WindowClass  # GUI

def main():
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
