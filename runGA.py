from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import QApplication,QGroupBox,QWidget, QVBoxLayout,QHBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QLineEdit,QFormLayout, QDialogButtonBox
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QInputDialog
from PyQt5.QtWidgets import QApplication, QLabel, QInputDialog
import cv2
from matplotlib import image as mpimg
from gaussian import gaussian_filter
from numpy import size, multiply, true_divide, array, zeros, ones, max, append
from numpy.matlib import repmat
import numpy as np
from GA import BaseGA
from PIO import BasePIO
from numpy.core.defchararray import find
from matplotlib import pyplot as plt
# data = cv2.imread('examples/evolutionary_based/b.jpg',cv2.IMREAD_GRAYSCALE)
from PyQt5.QtGui import QPixmap

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "pigeon"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("log.jpg"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self
                         .width, self.height)
        self.setStyleSheet("background-color:#059BC0")

        g1=QGroupBox("louzi")
        g2 = QGroupBox("fouzi")
        g3 = QGroupBox("Imagerie Médicale Du Sein")


        hbox1=QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()


        vbox = QVBoxLayout()
        self.btn1 = QPushButton("Importer Image")
        self.btn1.setIcon(QtGui.QIcon(""))
        self.btn1.setIconSize(QtCore.QSize(40,40))
        self.btn1.setToolTip("Cliquez ici")
        self.btn1.setStyleSheet("background-color:violet")
        self.btn1.clicked.connect(self.getImage)
        vbox.addWidget(self.btn1)

        self.nameLineEdit1 = QLineEdit()
        self.nameLineEdit2 = QLineEdit()
        self.nameLineEdit3 = QLineEdit()
        self.nameLineEdit4 = QLineEdit()
        self.nameLineEdit5 = QLineEdit()
        self.label=QLabel(self)
        self.label1 = QLabel("Nombre d'itérations:")
        self.label2 = QLabel("Nombre de la population :")
        self.label3 = QLabel("Nombre des Cluster  :")
        self.label4 = QLabel("Nombre de colomnes :")
        self.label5 = QLabel("Nombre de lignes :")
        hbox1.addWidget(self.label1)
        hbox1.addWidget(self.nameLineEdit1)

        hbox1.addWidget(self.label2)
        hbox1.addWidget(self.nameLineEdit2)

        hbox1.addWidget(self.label3)
        hbox1.addWidget(self.nameLineEdit3)

        hbox2.addWidget(self.label4)
        hbox2.addWidget(self.nameLineEdit4)

        hbox2.addWidget(self.label5)
        hbox2.addWidget(self.nameLineEdit5)
        hbox3.addWidget(self.label)
        g1.setLayout(hbox1)
        g2.setLayout(hbox2)
        g3.setLayout(hbox3)
        g1.setStyleSheet("background-color:")
        g3.setGeometry(0,0,10,10)

        vbox.addWidget(g1)
        vbox.addWidget(g2)
        vbox.addWidget(g3)



        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok )
        self.buttonBo = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.getInfo)
        self.buttonBo.rejected.connect(self.reject)
        self.buttonBox.setToolTip("Cliquez ici")
        self.buttonBo.setToolTip("Cliquez ici")
        self.buttonBox.setStyleSheet("background-color:green")
        self.buttonBo.setStyleSheet("background-color:red")
        vbox.addWidget(self.buttonBox)
        vbox.addWidget(self.buttonBo)
        self.setLayout(vbox)

        self.show()

    def getImage(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file','')
        self.imagePath = self.fname[0]
        pix=QPixmap(self.imagePath)
        self.label.setPixmap(QPixmap(pix))
        self.resize(pix.width(),pix.height())

    def reject(self):
        self.close()
    def getInfo(self):
        self.epoch = int(self.nameLineEdit1.text())
        self.pop_size = int(self.nameLineEdit2.text())
        self.dim = int(self.nameLineEdit3.text())
        self.col = int(self.nameLineEdit4.text())
        self.row = int(self.nameLineEdit5.text())



        img = cv2.imread(self.imagePath)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("image", img)

        gaussian3x3 = gaussian_filter(img, 3, sigma=1)

        cv2.imshow("gaussian", gaussian3x3)

        data = gaussian3x3
        #data=img
        o=0
        for i in range(0,data.shape[0]):
            for j in range(0, data.shape[1]):
                if (data[i][j] != 0):
                    o=o+1
        print(o)



        # data = img.reshape(img.shape[0] * img.shape[1], img.shape[2]).T
        maxX, maxY = data.shape[0], data.shape[1]




        def obj_func(X):
            # data = cv2.imread('examples/evolutionary_based/b.jpg',cv2.IMREAD_GRAYSCALE)
            #   img = mpimg.imread('b.jpg')

            # data = img.reshape(img.shape[0] * img.shape[1], img.shape[2]).T
            X = list(X)
            IMM = []
            ree1 = []
            u = []
            v = []
            c = []
            J = []
            distance2 = []
            IM = data
            # IM = float(data)
            maxX, maxY = data.shape[0], data.shape[1]
            N = 1

            Nc = size(X, 0) # nbr de ligne
            ree = repmat(0.000001, maxX, maxY)

            for k in range(0, Nc):
                IMM.append(IM)
                ree1.append(ree)

            for ag in range(0, N):
                for k in range(0, Nc):
                    v.append(X[k])
                    c.append(repmat(v[k], maxX, maxY))

                distance = array(IMM) - array(c)
                distance = multiply(distance, distance) + ree1
                invdistance = true_divide(1., distance)
                S = 0
                for k in range(0, Nc):
                    S = S + invdistance[:][:][k]

                for k in range(0, Nc):
                    distance2.append(multiply(distance[:][:][k], S))
                    u.append(1. / distance2[:][:][k])
                J1 = 0
                for k in range(0, Nc):
                    mult = multiply(u[:][:][k], u[:][:][k])
                    J1 = J1 + sum(sum(multiply(mult, distance[:][:][k])))
                J.append(J1)
            return array(J)
        ## Setting parameters
        verbose = False

        # A - Different way to provide lower bound and upper bound. Here are some examples:

        ## 1. When you have different lower bound and upper bound for each parameters
        LB_ = list(zeros(self.dim))
        UB_ = multiply(256, ones(self.dim)).tolist()

        # --------------------------------------- PIO

        md2 = BaseGA(obj_func, LB_, UB_, verbose, self.epoch, self.pop_size)
        best_pos2, best_fit2, list_loss2 = md2.train

        print('--------------------PIO-------------------')
        print(md2.solution)
        print(best_pos2)
        print(best_fit2)
        print(list_loss2)


        plt.style.use('ggplot')
        plt.ylabel('Fitnesse')
        plt.xlabel("nombre d'itérations")
        plt.title("graphe d'évolution de la fonction Fitnesse")
        ff="Nombre de la population="+str(self.pop_size)
        plt.plot(list_loss2,label=ff)
        plt.legend()
        plt.style.use('default')


        centers = sorted(list(best_pos2))
        Nc = self.dim
        maxX, maxY = data.shape[0], data.shape[1]
        ree = repmat(0.000001, maxX, maxY)
        IMM = np.zeros(shape=(maxX, maxY, Nc))
        ree1 = np.zeros(shape=(maxX, maxY, Nc))
        c = np.zeros(shape=(maxX, maxY, Nc))

        SommeInvDist2 = []
        for k in range(0, Nc):
            IMM[:, :, k] = data
            ree1[:, :, k] = ree

        for k in range(0, Nc):
            c[:, :, k] = np.tile(centers[k], (maxX, maxY))

        distance = (IMM) - (c)
        distance = (distance * distance) + ree1
        invdistance = 1 / distance
        SommeInvDist = np.sum(invdistance, 2)
        SommeInvDist2 = np.zeros(shape=(maxX, maxY, Nc))
        for k in range(0, Nc):
            SommeInvDist2[:, :, k] = SommeInvDist

        distance2 = (distance * SommeInvDist2)
        u = 1 / distance2
        print(u.shape)
        U = np.zeros(shape=(Nc, maxX * maxY))
        for k in range(Nc):
            U[k, :] = np.reshape(u[:, :, k], (1, (maxX * maxY)))

        maxU = U.max(0)
        index = []
        for k in range(Nc):
            index.append(np.where(U[k, :] == maxU))

        fcmImage = np.zeros(shape=(1, maxX * maxY))
        for k in range(Nc):
            fcmImage[:, index[k]] = centers[k]


        imagNew = np.reshape(fcmImage, (maxX, maxY))
        plt.figure()

        plt.imshow(imagNew, cmap='gray', vmin=0, vmax=255, interpolation='none',)
        print(self.dim)

        plt.figure()
        for a  in range(0,self.dim):


            fcmImage = np.zeros(shape=(1, maxX * maxY))
            fcmImage[:, index[a]] = np.uint8(centers[a])
            imagNew = np.reshape(fcmImage, (maxX, maxY))
            s = 0

            for i in range(0, maxX):
                for j in range(0, maxY):

                    if (imagNew[i][j] == int(max(centers))):

                        s = s + 1

            print(s)
            mas=100*(s/o)
            print(mas , '%')

            plt.subplot(self.col, self.row, a+1,title="la masse" + str(round(mas,2)) +  "%" )
            plt.subplots_adjust(wspace=0.3,hspace=0.4)
            plt.axis('off')



            plt.imshow(imagNew, cmap='gray', vmin=0, vmax=255, interpolation='none')


        plt.show()





App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
