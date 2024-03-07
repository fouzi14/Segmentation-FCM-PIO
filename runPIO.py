from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import QApplication,QGroupBox,QWidget,QErrorMessage, QVBoxLayout,QHBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QLineEdit,QFormLayout, QDialogButtonBox
import sys
from PIL import Image
from PyQt5.QtGui import QFont, QRegExpValidator
from PyQt5.QtWidgets import QApplication, QLabel, QInputDialog
from PyQt5.QtWidgets import QApplication, QLabel, QInputDialog
import cv2
from matplotlib import image as mpimg
from gaussian import gaussian_filter
from numpy import size, multiply, true_divide, array, zeros, ones, max, append
from numpy.matlib import repmat
import numpy as np


from PIO import PIO
from numpy.core.defchararray import find
from matplotlib import pyplot as plt
# data = cv2.imread('examples/evolutionary_based/b.jpg',cv2.IMREAD_GRAYSCALE)
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *



class Window(QWidget):


    table_figure=[None] #nhoto fih l'image segmenter
    plotta3na=[None] #nhoto fih graphe
    def __init__(self):
        super().__init__()

        self.title = "pigeon"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.InitWindow()



    def image3(self): # ta3 l'affichage l'image segmenter n3ayrtolha bl boutan b3
        plt.figure()
        plt.imshow(self.table_figure[0], cmap='gray', vmin=0, vmax=255, interpolation='none')








    def image4(self): #fonction t afichilna les diffirant cluster n3aytolha bl boutan b4


        plt.figure()
        for a in range(0,self.dim):
            if a<self.dim-1:
                # mat2afichilnache volume de la masse fi les cluster limafihomche tumer
                plt.subplot(self.col, self.row, a+1)
            else:
                # t2affichi volume ghil fl cluster ta3 la masse
                plt.subplot(self.col, self.row, a + 1, title="la masse" + str(round(self.mas, 2)) + "%")

            plt.subplots_adjust(wspace=1, hspace=0.4)
            plt.axis('off')
            #n2aficho les diffirants classes ta3 l'image
            plt.imshow(self.table_f[a], cmap='gray', vmin=0, vmax=255, interpolation='none')
            #n2aficho biha lgraphe bl boutan b5
    def image5(self):
        plt.style.use('ggplot')
        plt.ylabel('Fitnesse')
        plt.xlabel("nombre d'itérations")
        plt.title("Graphe d'évolution de la fonction Fitnesse")
        ff = "Nombre de la population=" + str(self.population)

        plt.plot(self.plotta3na[0], label=ff)
        plt.legend()


    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("log.jpg"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("QPushButton{background-color:rgb(225,225,225);border:1px solid grey;padding:5px;}QPushButton:hover{background-color:#778899;border:1px solid grey;color:white;}")


        g1=QGroupBox("Paramètre d'éxécution")
        g1.setFont(QtGui.QFont('SansSerif',10))

        g2 = QGroupBox("Paramètre d'affichage")
        g2.setFont(QtGui.QFont('SansSerif',10))
        g3 = QGroupBox("Imagerie Médicale Du Sein")
        g3.setFont(QtGui.QFont('SansSerif', 10))
        g4 = QGroupBox("Imagerie traité")
        g4.setFont(QtGui.QFont('SansSerif', 10))
        g5 = QGroupBox("Description")
        g5.setFont(QtGui.QFont('SansSerif', 10))

        hbox1=QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        self.des = QLabel("La masse tumorale est")
        self.m =QLabel(self)  # n7oto fiha  % ta3 la masse

        hbox5 = QHBoxLayout()
        hbox5.addWidget(self.des)
        hbox5.addWidget(self.m)


        vbox = QVBoxLayout()
        self.btn1 = QPushButton("Importer Image")
        #self.btn1 .setFont(QtGui.QFont('SansSerif', 13))
        #self.btn1.setIcon(QtGui.QIcon("log.jpg"))
        self.btn1.setIconSize(QtCore.QSize(40,40))
        self.btn1.setToolTip("Cliquez ici Pour importés l'image")


        self.btn1.clicked.connect(self.getImage)
        vbox.addWidget(self.btn1)

        self.nameLineEdit1 = QLineEdit("20")
        self.nameLineEdit2 = QLineEdit("100")
        self.nameLineEdit3 = QLineEdit("5")
        self.nameLineEdit4 = QLineEdit("2")
        self.nameLineEdit5 = QLineEdit("3")


        reg=QRegExp("[0-9]+[0-9]") # bah mataa3tinach la man bah nakatbo des caractaire
        validator1=QRegExpValidator(reg,self.nameLineEdit1)
        self.nameLineEdit1.setValidator(validator1)
        validator2 = QRegExpValidator(reg, self.nameLineEdit2)
        self.nameLineEdit2.setValidator(validator2)
        validator3 = QRegExpValidator(reg, self.nameLineEdit3)
        self.nameLineEdit3.setValidator(validator3)
        validator4 = QRegExpValidator(reg, self.nameLineEdit4)
        self.nameLineEdit4.setValidator(validator4)
        validator5 = QRegExpValidator(reg, self.nameLineEdit5)
        self.nameLineEdit5.setValidator(validator5)

        self.label=QLabel(self) # bah n7oto fiha image path


        self.label1 = QLabel("Nombre d'itérations:")
        self.label2 = QLabel("Nombre de la population :")
        self.label3 = QLabel("Nombre des Cluster  :")

        self.label4 = QLabel("Nombre de lignes :")
        self.label5 = QLabel("Nombre de colomnes :")
        hbox1.addWidget(self.label1)
        hbox1.addWidget(self.nameLineEdit1)
        self.nameLineEdit1.setStyleSheet("background-color:#FFFFFF")

        hbox1.addWidget(self.label2)
        hbox1.addWidget(self.nameLineEdit2)
        self.nameLineEdit2.setStyleSheet("background-color:#FFFFFF")

        hbox1.addWidget(self.label3)
        hbox1.addWidget(self.nameLineEdit3)
        self.nameLineEdit3.setStyleSheet("background-color:#FFFFFF")

        hbox2.addWidget(self.label4)
        hbox2.addWidget(self.nameLineEdit4)
        self.nameLineEdit4.setStyleSheet("background-color:#FFFFFF")

        hbox2.addWidget(self.label5)
        hbox2.addWidget(self.nameLineEdit5)
        self.nameLineEdit5.setStyleSheet("background-color:#FFFFFF")


        hbox3.addWidget(self.label)
        self.l = QLabel(self) # n7oto fiha image filtré
        hbox4.addWidget(self.l)

        self.l2 = QLabel(self) # nhoto fiha l'image segmenter
        hbox4.addWidget(self.l2)


        self.l3 = QLabel(self)#nhoto fiha l'image ta3 la masse
        hbox4.addWidget(self.l3)





        self.b3=QPushButton("Image segmenté")
        hbox4.addWidget(self.b3)
        self.b3.clicked.connect(self.image3)
        self.b4=QPushButton("les cluster")
        hbox4.addWidget(self.b4)
        self.b4.clicked.connect(self.image4)
        self.b5=QPushButton("Graphe d'evol")
        hbox4.addWidget(self.b5)
        self.b5.clicked.connect(self.image5)

        self.b3.setToolTip("Cliquez ici Pour afficher l'image segmenté")
        self.b4.setToolTip("Cliquez ici Pour afficher les defirent cluster d'image segmenté")
        self.b5.setToolTip("Cliquez ici Pour afficher Graphe d'evolution de la fonction fitness")


        g1.setLayout(hbox1)
        g2.setLayout(hbox2)
        g3.setLayout(hbox3)
        g4.setLayout(hbox4)
        g5.setLayout(hbox5)




        vbox.addWidget(g1)
        vbox.addWidget(g2)
        vbox.addWidget(g3)
        vbox.addWidget(g4)
        vbox.addWidget(g5)

        self.re = QPushButton("Reset")
        self.buttonBox = QPushButton("Ok")
        self.buttonBo = QPushButton("Quitter")
        self.buttonBox.clicked.connect(self.getInfo)
        self.buttonBo.clicked.connect(self.reject)
        self.re.clicked.connect(self.res)
        self.buttonBox.setToolTip("Cliquez ici Pour lancer le traitement")
        self.buttonBox.setStyleSheet("width:100%;")
        self.buttonBo.setStyleSheet("width:50px;")
        self.buttonBo.setToolTip("Cliquez ici Pour arrêter le traitement")

        vbox.addWidget(self.buttonBox)
        vbox.addWidget(self.buttonBo)
        vbox.addWidget(self.re)
        self.setLayout(vbox)

        self.show()

    def getImage(self):


        self.fname = QFileDialog.getOpenFileName(self, 'Open file','')
        self.imagePath = self.fname[0]



        pix=QPixmap(self.imagePath)



        self.label.setPixmap(QPixmap(pix))
        #self.resize(pix.width(),pix.height())
        self.l.setPixmap(QPixmap(None))
        self.l2.setPixmap(QPixmap(None))
        self.l3.setPixmap(QPixmap(None))

        self.m.setText(None)


    def res(self):
        self.l.setPixmap(QPixmap(None))
        self.l2.setPixmap(QPixmap(None))
        self.l3.setPixmap(QPixmap(None))

        self.m.setText(None)


    def reject(self):
        self.close()


    def getInfo(self):



        self.étiration = int(self.nameLineEdit1.text())
        self.population = int(self.nameLineEdit2.text())
        self.dim = int(self.nameLineEdit3.text())
        self.col = int(self.nameLineEdit4.text())
        self.row = int(self.nameLineEdit5.text())


        try:

            img = cv2.imread(self.imagePath)
        # cv2.imshow("image", img)


            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            gaussian3x3 = gaussian_filter(img, 3, sigma=1)


            imageGAUSS = Image.fromarray(gaussian3x3)

            imageGAUSS.save("save\gaussian.png")
            p = QPixmap("save\gaussian.png")

            self.l.setPixmap(QPixmap(p))
            self.resize(p.width(), p.height())
            # cv2.imshow("gaussian", gaussian3x3)

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




            def obj_func(position):

                position = list(position)
                IMM = []
                ree1 = []
                u = []
                v = []
                c = []
                J = []
                distance2 = []
                IM = data

                maxX, maxY = data.shape[0], data.shape[1]
                N = 1

                Nc = size(position, 0) # nbr de ligne
                #nsan3o matrice nmbr de ligne maxX w nmbre de colomne maxY wl epsilen 0.0000000001
                ree = repmat(0.0000000001, maxX, maxY)

                for k in range(0, Nc):
                    #n3amro liste ta3 IMM bles matrice ta3 data 3la hsab le nmbre de cluster
                    IMM.append(IM)
                    #n3amro liste ta3 ree1 bles matrice ta3 epsilent 3la hsab nmbre de luster
                    ree1.append(ree)

                for ag in range(0, N):
                    for k in range(0, Nc):
                    # v : liste n3amro fiha les solustion (position)
                        v.append(position[k])
                    # c : liste n3aamro fiha les matrice ta3 v[k]
                        c.append(repmat(v[k], maxX, maxY))
                    # nhasbo la distence bin les pixel wl les centre
                    distance = array(IMM) - array(c)
                    distance = multiply(distance, distance)
                    invdistance = true_divide(1., distance)
                    S = 0
                    for k in range(0, Nc):
                        S = S + invdistance[:][:][k]

                    for k in range(0, Nc):
                        distance2.append(multiply(invdistance[:][:][k], S))
                        u.append(1. / distance2[:][:][k])
                    J1 = 0
                    for k in range(0, Nc):
                        mult = multiply(u[:][:][k], u[:][:][k])
                        J1 = J1 + sum(sum(mult))


                    J.append(J1)

                return array(J)
            ## Setting parameters
            verbose = False




            LB_ = list(zeros(self.dim))
            UB_ = multiply(255, ones(self.dim)).tolist()

            # --------------------------------------- PIO

            md2 = PIO(obj_func, LB_, UB_, verbose, self.étiration, self.population)
            best_pos2, best_fit2, list_loss2 = md2.train

            print('--------------------PIO-------------------')
            print(md2.solution)
            print(best_pos2)
            print(best_fit2)
            print(list_loss2)


            self.plotta3na[0] = list_loss2


            centers = sorted(list(best_pos2))
            Nc = self.dim
            maxX, maxY = data.shape[0], data.shape[1]
            ree = repmat(0.000000001, maxX, maxY)
            IMM = np.zeros(shape=(maxX, maxY, Nc))
            ree1 = np.zeros(shape=(maxX, maxY, Nc))
            c = np.zeros(shape=(maxX, maxY, Nc))


            for k in range(0, Nc):
                IMM[:, :, k] = data
                ree1[:, :, k] = ree

            print(data.shape)

            for k in range(0, Nc):
                c[:, :, k] = np.tile(centers[k], (maxX, maxY))

            distance = (IMM) - (c)
            distance = (distance * distance) + ree1
            invdistance = 1 / distance
            SommeInvDist = np.sum(invdistance)
            SommeInvDist2 = np.zeros(shape=(maxX, maxY, Nc))
            for k in range(0, Nc):
                SommeInvDist2[:, :, k] = SommeInvDist

            distance2 = (distance * SommeInvDist2)
            u = 1 / distance2
            print( u.shape)
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

            # image segmente
            imagNew = np.reshape(fcmImage, (maxX, maxY))


            imageseg = Image.fromarray(imagNew)

            imageseg.convert('RGB').save("save\seg.png") #save image
            p = QPixmap("save\seg.png") #l'apel de l'image

            self.l2.setPixmap(QPixmap(p)) #nhotoha fl l2
            self.resize(p.width(), p.height())
            self.table_figure[0]=imagNew #nhotoha fl table de figure




            print(self.dim)
            self.table_f = [None, None, None, None, None,None,None,None,None,None] #table nhoto fih les image ta3 cluster

            for a  in range(0,self.dim):


                fcmImage = np.zeros(shape=(1, maxX * maxY))
                fcmImage[:, index[a]] = np.uint8(centers[a]) # fcmimage rahi tcacifer fi le
                imagNew = np.reshape(fcmImage, (maxX, maxY))
                imagemass = Image.fromarray(imagNew)

                imagemass.convert('RGB').save("save\masse.png")
                p = QPixmap("save\masse.png")


                self.l3.setPixmap(QPixmap(p))
                self.resize(p.width(), p.height())

                self.table_f[a] = imagNew

                s = 0
                ma= ""

                for i in range(0, maxX):
                    for j in range(0, maxY):

                        if (imagNew[i][j] == int(max(centers))):
                            s = s + 1



                print(s)


                self.a=ma
                self.mas=100*(s/o)
                self.mx=self.mas
                self.m.setText(str(round(self.mas, 2)) + "%")
                print(self.mas , '%')

            plt.show()

        except:
            print("fighfighvgjohhgjh")
            eror = QErrorMessage()
            eror.showMessage("importer l'imagerie stp !")
            eror.setWindowIcon(QtGui.QIcon("log.jpg"))
            eror.setWindowTitle(self.title)

            eror.exec()

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())