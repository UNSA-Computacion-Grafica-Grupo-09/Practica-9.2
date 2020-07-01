

import cv2 as cv
import numpy as np
#import imutils
import math

def rotar_imagen(image):
    img = image.copy()
    ancho = img.shape[1]  #Columnas
    alto = img.shape[0] #Filas
    
    # def matrix_rotate(angulo, tx, ty):
    # angulo = angulo de rotcion en rdianes 
    # tx = el centro de la imagen en x
    # ty = el centro de la imagen en y
    
    #Matriz de rotación

    def matrix_rotacion(img, angle):
        h, w = img.shape[:2]
        img_c = (w / 2, h / 2)
        rad = math.radians(angle)
        seno = math.sin(rad)
        coseno = math.cos(rad)
        b_w = int((h * abs(seno)) + (w * abs(coseno)))
        b_h = int((h * abs(coseno)) + (w * abs(seno)))
        mid_h = int((h+1)/2)
        mid_w = int((w+1)/2)
        A = np.float32([[coseno,seno],[-seno,coseno]])
        B = [[((1-coseno)*mid_w)-(seno*mid_h)],[(seno*mid_w)+(1-seno)*mid_h]]
        M = np.concatenate((A, B), axis=1)
        M[0, 2] += ((b_w / 2) - img_c[0])
        M[1, 2] += ((b_h / 2) - img_c[1])
        return M


    def Affines(image,M,dim ) :
            fil , col = image.shape [0:2]#tomamos las dimensiones
            #img_res = np.zeros ([ dim[1],dim[0],3] , dtype = np . uint8 )
            image_res=np.zeros((fil,col,3),np.uint8)#Creamos nuestra matriz para la respuesta
            A = M [:2,:2]
            B = M [:,2:]
            for i in range ( fil ):#tomamos las dimensiones
                for j in range ( col ) :
                    res = np.dot(A,np.float32([[i],[j]]))+B #aplicamos la operacion A*M+B
                    res = np.uint32(res)
                    if res[0,0]>=dim[1] or res[1,0]>=dim[0]:
                        continue

                    image_res [res[0,0],res[1,0]] = image[i,j]#colocaos en su pixel correspondiente
            return image_res


    
    def AffineMejorado(img,M):
        X=[0,0]
        h,w,c=img.shape
        img_out=np.zeros((h,w,c),np.uint8)#Creamos nuestra matriz para la respuesta
        iden=np.array([[M[1][1],M[1][0]],[M[0][1],M[0][0]]])
        B=np.array([[M[1][2]],[M[0][2]]])
        for i in range(h):
            for j in range(w):#tomamos las dimensiones
                vector=np.array([[i],[j]])
                Y=vector-B
                X=cv.solve(iden,Y)#aplicamos la operacion solve A*Y+X
                x=int(X[1][0][0])
                y=int(X[1][1][0])
                if(x<img_out.shape[0] and x>=0):
                    if(y<img_out.shape[1] and y>=0):
                        
                        img_out[i][j]=img[x][y]#colocaos en su pixel correspondiente

                        
        return img_out

    M = matrix_rotacion(imagen1,30)
    #M = matrix_rotacion(0.30, ancho//2, alto//2)
    #imageOut = Affines(img, M , (ancho, alto))
    #imageOut = Affines(img, M , (ancho, alto))  
    imageOut = AffineMejorado(img, M) 
        
    return imageOut

imagen1 = cv.imread("orig.png")
result = rotar_imagen(imagen1)
cv.imwrite("Affine_ROTAPRUEBAsds.jpg", result)
cv.waitKey(0)