import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math as mt

def AffineTransform(pts1,pts2):
    M = []
    for i in range(2):
        A = []
        for j in range(len(pts1)):
            A_ = [pts1[j][0],pts1[j][1],1]
            A.append(A_)
        B = [pts2[0][i],pts2[1][i],pts2[2][i]]
        X = np.linalg.inv(A).dot(B)
        M.append([X[0],X[1],X[2]])
    return M


def AffineMejorado(M,img,hei,wei):
    X=[0,0]
    h,w,c=img.shape
    img_out=np.zeros((hei,wei,c),np.uint8)#Creamos nuestra matriz para la respuesta
    iden=np.array([[M[1][1],M[1][0]],[M[0][1],M[0][0]]])
    B=np.array([[M[1][2]],[M[0][2]]])
    for i in range(hei):#tomamos las dimensiones
        for j in range(wei):
            vector=np.array([[i],[j]])#aplicamos la operacion solve(A,Y,X)
            Y=vector-B
            X=cv.solve(iden,Y)
            x=int(X[1][0][0])
            y=int(X[1][1][0])
            if(x<img_out.shape[0] and x>=0):
                if(y<img_out.shape[1] and y>=0):
                    
                    img_out[i][j]=img[x][y]#colocaos en su pixel correspondiente

                    
    return img_out

img = cv.imread('ejer2.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv.getAffineTransform(pts1,pts2)
#M = cv.getAffineTransform(pts1,pts2)
M = AffineTransform(pts1,pts2)
#M = matrix_rotacion(0.261799, rows//2, cols//2)
#print(M.shape)
#dst = cv.warpAffine(img,M,(cols,rows))

#dst = warpaffine(img,M)
dst = AffineMejorado(M,img,rows,cols)
#imagen1 = cv.imread("marvel.jpg")
cv.imwrite("Mejor_Affine_PUNTOS.jpg", dst)
#plt.subplot(121),plt.imshow(img),plt.title('Input')
#plt.subplot(122),plt.imshow(dst),plt.title('Output')
#plt.show()