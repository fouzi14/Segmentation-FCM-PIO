# importing libraries
from PyQt5.QtWidgets import *
import sys

# creating a class
# that inherits the QDialog class
class Window(QDialog):

	def __init__(self):
		super(Window, self).__init__()
		self.setGeometry(100, 100, 300, 400)
		self.formGroupBox = QGroupBox("Form 1")
		self.ageSpinBar = QSpinBox()
		self.degreeComboBox = QComboBox()
		self.nameLineEdit = QLineEdit()
		self.createForm()
		self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		self.buttonBox.accepted.connect(self.getInfo)
		self.buttonBox.rejected.connect(self.reject)
		mainLayout = QVBoxLayout()
		mainLayout.addWidget(self.formGroupBox)
		mainLayout.addWidget(self.buttonBox)

		self.setLayout(mainLayout)
	def getInfo(self):
		print("Person Name : {0}".format(self.nameLineEdit.text()))
		self.close()

	def createForm(self):
		layout = QFormLayout()
		layout.addRow(QLabel("Name"), self.nameLineEdit)

		self.formGroupBox.setLayout(layout)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec())
