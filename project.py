from collections import Counter
from statistics import stdev
from math import *
import re
from typing_extensions import Self
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation

atributosEntrenamiento, setEntrenamiento, setPrueba, habilitados = [], [], [], []
datasetE = 'datasets/sb1-T.txt'; datasetP = 'datasets/sb1-P.txt'

def leerT():  #Lectura del set de Entrenamiento.
    with open(datasetE, "r") as f:
        global NinstanciasEntrenamiento   #Instancias totales del conjunto de Entrena.
        NinstanciasEntrenamiento = int(f.readline())
        global NatributosEntrenamiento   #Atributos totales del conjunto de Entrena.
        NatributosEntrenamiento = int(f.readline())
        global atributosEntrenamiento   #Tipo de cada atributo del conjunto de Entrena (nominal/numerico) + clases.
        atributosEntrenamiento = f.readline()[:-1]
        atributosEntrenamiento = atributosEntrenamiento.split(",")
        for line in f:   #Lectura de cada instancia.
            line = line[:-1]
            setEntrenamiento.append(line.split(","))   #Datos las intancias en esta variable.
        f.close()

def leerP():  #Lectura del set de Prueba.
    with open(datasetP, "r") as f:
        global NinstanciasPrueba   #Instancias totales del conjunto de Prueba.
        NinstanciasPrueba = int(f.readline())
        global NatributosPrueba   #Atributos totales del conjunto de Prueba.
        NatributosPrueba = int(f.readline())
        global atributosPrueba   #Tipo de cada atributo del conjunto de Prueba (nominal/numerico). + clases.
        atributosPrueba = f.readline()[:-1]
        atributosPrueba = atributosPrueba.split(",")
        for line in f:   #Lectura de las instancias.
            line = line[:-1]
            setPrueba.append(line.split(","))   #Datos de las instancias en esta variable.
        f.close()

def contarNomi():  #Conteo de valores nominales para aplicar hvdm nominal.
    global conteototal
    conteototal = [dict()] * NatributosEntrenamiento  #Se crea un diccionario por cada atributo del dataset
    #Se crean n listas con x diccionarios donde n es el numero total de clases y x el numero...
    #... de atributos del dataset, para el conteo por clase.
    global conteoClase
    conteoClase = [[dict()] * NatributosEntrenamiento for _ in range(int(atributosEntrenamiento[-1]))]  
    for x in range(len(atributosEntrenamiento[:-1])):  #Recorremos cada uno de los atributos.
        #Se agregan los posibles valores de cada atributo, tanto para el conteo total como por cada clase.
        conteototal[x] = dict.fromkeys(range(int(atributosEntrenamiento[x])), 0)
        for y in range(int(atributosEntrenamiento[-1])):
            conteoClase[y][x] = dict.fromkeys(range(int(atributosEntrenamiento[x])), 0)
    for x in setEntrenamiento:  #Por cada instancia del set de entrenamiento.
        clase = int(x[-1])  #Obtenemos la clase de cada instancia.
        for y in range(len(x[:-1])):  #Recorremos cada uno de los atributos de dicha instancia.
            if(atributosEntrenamiento[y] != "0"):  #Si es atribuito nominal...
                conteototal[y][int(x[y])] += 1   #... Va haciendo conteo global...
                conteoClase[clase][y][int(x[y])] += 1  #...Y por clase.

def desviacion():  #Calculo de las desviaciones estandar en caso de atributos numericos.
    global desviaciones  #Se crea un diccionario por cada atributo del dataset.
    desviaciones = [dict()] * NatributosEntrenamiento; ls = []  #ls solo es para calcular la desviacion con stdev.
    for x in range(len(atributosEntrenamiento[:-1])):  #Por cada atributo del set.
        if atributosEntrenamiento[x] == '0':  #Si es un atributo numerico.
            for obj in setEntrenamiento:  #Almacenamos en ls los valores numericos de dicho atributo.
                ls.append(float(obj[x]))
            des = stdev(ls)  #Calculamos la desviacion estandar.
            desviaciones[x] = dict.fromkeys(['desv'], des)  #Agregamos la desviacion a su atributo correspondiente.

def clasificacion(hvdm, real):   #Metodo para la clasificacion de la instancia.
    cercania = []; pos = 0; ban = 1; min = 0   #Variables necesarias.
    for x in hvdm:  #Ciclo para juntar las distancias de acuerdo a su clasificacion.
        ban = 1
        if(len(cercania)):
            for j in cercania:
                if(j[1] == x[1]):
                    j = list(j)
                    j[0] = j[0]+x[0]
                    j = tuple(j)
                    cercania[pos] = j
                    ban = 0
                else:
                    pos+=1
            pos = 0
            if(bool(ban)):
                cercania.append(x)
        else:
            cercania.append(x)
    clases = Counter(elem[1] for elem in hvdm)  #Conteo de clases de los kvecinos mas cercanos.
    clases = clases.items()   #Convertimos a un diccionario.
    # --- SI SE QUIEREN VER LAS CLASES MAS FRECUENTES DESCOMENTAR LA LINEA CONSECUENTE ---
    # print(clases)  #La interpretacion es por tuplas donde el primer valor es la clase y el segundo la frecuencia.
    cer = cercania[0][0]  #Se asigna esta variable a la primera distancia conjunta de los kvecinos (fines de funcionamiento).
    for x in clases:  #Empezamos a analizar la clase mas frecuente y/o mas cercana.
        if(x[1]>min):  #Si es mas frecuente se establece como la que va a clasificar.
            minor = x
            min = x[1]
        elif(x[1]==min):  #Si tienen la misma frecuencia se determina en base a su distancia (mas baja gana).
            for bus in cercania:
                if(bus[1] == x[0]):
                    distancia = bus[0]
                    break;
            if(distancia < cer):
                minor = x
                min = x[1]
                cer = distancia
    # --- SI SE QUIERE VER LA CLASIFICACION ELEGIDA POR EL ALGORITMO Y LA REAL DESCOMENTAR LA LINEA CONSECUENTE. ---
    # print("Algorithm class: "+str(minor[0])+" | Real class: "+str(real)+" | Frecuency: "+str(minor[1]))
    if(int(minor[0] == real)): #Aumenta la variable "correctas" si la clasificacion hecha por el algoritmo es correcta.
        global correctas
        correctas += 1
    global totales  #Aumentamos el numero de instancias totales clasificadas por el algoritmo.
    totales += 1

def kNeighbors (tam):  #Procedimiento del algoritmo KNN basado en distancia HVDM.
    global correctas   #Declaramos las variables de conteo de clasificacion a 0 cada que corremos kNeighbors.
    correctas = 0
    global totales
    totales = 0
    eva = []  #Lista que contiene el hvdm de cada instancia del set de Entrenamiento por cada instancia de prueba.
    
    for instancia in setPrueba:  #Por cada instancia del set de prueba.
        for instanciaEntre in setEntrenamiento:  #Se evalua por cada instancia del set de entrenamiento.
            hvdm = 0
            for i in range(len(instancia[:-1])):  #Recorremos cada atributo de la instancia.
                if(habilitados[i] != 0):  #Siempre y cuando dicho atributo esta habilitado para calcular hvdm.
                    if (atributosEntrenamiento[i] != '0'):  #Si es atributo nominal.
                        hvdmnomi = 0
                        for j in range(int(atributosEntrenamiento[-1])):  #Calculamos HVDM nominal.
                            hvdmnomi += (conteoClase[j][i][int(instanciaEntre[i])]/conteototal[i][int(instanciaEntre[i])]) - (conteoClase[j][i][int(instancia[i])]/conteototal[i][int(instancia[i])])
                        hvdm += (pow(abs(hvdmnomi), 2))  #Vamos almacenando los resultados respectivos.
                    else:  #Si es atributo numerico, calculamos HVDM numerico y se alamcena el resultado de operacion.
                        hvdm += pow(abs((float(instanciaEntre[i]) - float(instancia[i])))/(4*desviaciones[i]['desv']), 2)
            eva.append((sqrt(hvdm), instanciaEntre[-1]))  #Guardamos el HVDM final obtenido de esta instancia del entrenamiento.
        order = sorted(eva)  #Ordenamos las distancias HVDM encontradas ascendentemente.
        kvecinos2 = order[:tam]  #Nos quedamos con los k distancias mas cercanas (k vecinos).
        # --- SI SE QUIERE VER LA INSTANCIA DE PRUEBA A CLASIFICAR DESCOMENTAR LA LINEA CONSECUENTE. ---
        # print("-----------------\n"+str(instancia))
        clasificacion(kvecinos2, instancia[-1])  #Hacemos la clasificacion de la instancia de prueba.
        order.clear(); eva.clear(); kvecinos2.clear()  #Limpiamos la variables del algoritmo correspondientes.
    result = (correctas*100.0)/totales  #Regla de 3 para obtener el resultado final de clasificacion en forma de %.
    #Mostramos los resultados finales del algoritmo.
    print("\n -> Resultado de clasificacion: %"+str(result)+"\n -> Total de instancias clasificadas: "+str(totales)+"\n -> Instancias clasificadas correctamente: "+str(correctas))

def convertarff(dataset):  #Convertir la sintaxis del dataset a formato .arff para Weka.

    # --- No hay mucho que comentar de este metodo, simplemente se le asigna un nombre a cada...
    # atributo del dataset en cuestion y mediante manipulacion de cadenas se crea el archivo. ---

    f = open("datasetarff.arff", mode="w")
    f.write("@relation archivo.convertido.arff\n\n")

    cad = "@attribute "; temp = ""
    cont = 0
    for i in atributosEntrenamiento[:-1]:
        if(i != '0'):
            for k in range(0, int(i), 1):
                temp += str(k)
                if (k+1 != int(i)):
                    temp += ", "
            f.write(cad+"Atr"+str(cont)+" {"+temp+"}\n")
            cont += 1
            temp = ""
        else:
            f.write(cad+"Atr"+str(cont)+" REAL\n")
            cont += 1
    clase = atributosEntrenamiento[-1]
    for i in range(0, int(clase), 1):
        temp += str(i)
        if(i+1 != int(clase)):
            temp += ", "
    f.write(cad+"Clase"+" {"+temp+"}\n\n@data\n")

    for i in dataset:
        for j in i[:-1]:
            f.write(j+",")
        f.write(i[-1]+"\n")
    f.close()

def arbol(kvecinos):  #Creacion de un arbol J48 de Weka y ejecutar KNN en base a los atributos del mismo.
    convertarff(setEntrenamiento)  #Cambio de sintaxis a formato .arff del set de entrenamiento.
    poda = []  #Lista que almacena que atributos hechos por el arbol.
    loader = Loader(classname="weka.core.converters.ArffLoader")   #Preparamos la lectura del archivo .arff
    datasetarff = loader.load_file("datasetarff.arff")   #Cargamos dicho dataset.
    datasetarff.class_is_last()   #Establecemos que el ultimo atributo corresponde a la clase.
    #Clasificador J48 + opciones para crear arbol podado.
    clasificador = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"]) 
    #Clasificador J48 + opciones para crear el arbol sin podar.
    clasificador2 = Classifier(classname="weka.classifiers.trees.J48", options=["-U"])  #Clasificador utilizado + opciones.
    evaluation = Evaluation(datasetarff)  #Necesario para evaluar el arbol podado y obtener la matriz de confusion.
    evaluation2 = Evaluation(datasetarff)  #Necesario para evaluar el arbol completo y obtener la matriz de confusion.
    #Construimos los arboles en base al dataset de entrenamiento.
    clasificador.build_classifier(datasetarff)  
    clasificador2.build_classifier(datasetarff)
    print(clasificador)  #Mostramos el arbol podado construido.
    arbolito = clasificador.graph  #Del arbol podado obtenemos el grafo en texto del cual extraeremos los atributos usados.
    for line in arbolito.splitlines():  #Por cada linea del arbol.
        ext = re.findall('"([^"]*)"', line)  #Guardamos unicamente el texto que tenga entre comillas.
        if(len(ext) != 0):  #Si el texto no esta vacio...
            if("Atr" in ext[0]):   #... Y contiene "Atr"
                take = ext[0][3:]  #Guardamos el numero del atributo.
                if(int(take) in poda):  #Si dicho atributo ya se encuentra en lista almacen omitimos...
                    pass
                else:  #... Caso contrario lo añadimos.
                    poda.append(int(take))  
        # --- SI SE QUIERE VISUALIZAR EL GRAFO EN TEXTO DESCOMENTAR LA LINEA CONSECUENTE. ---
        # print(line+" ... "+str(ext), len(ext))
    poda.sort()  #Ordenamos los numeros de los atributos encontrados para una mejor visualizacion.
    # --- SI SE QUIERE OBSERVAR QUE ATRIBUTOS SON LOS USO EL ARBOL DESCOMENTAR LA LINEA CONSECUENTE. ---
    print("\n=== Atributos utilizados del arbol podado ===\n"+str(poda)+"\n")
    for i in range(0, len(habilitados), 1):  #Modificamos los atributos habilitados para KNN.
        if(i in poda):  #Si el atributo se encuentra dentro de los podados lo deja habilitado.
            habilitados[i] = 1
        else:  #Caso contrario lo deshabilita.
            habilitados[i] = 0
    convertarff(setPrueba)  #Cambio de sintaxis a formato .arff del set de prueba.
    datasetarff = loader.load_file("datasetarff.arff")   #Cargamos el dataset.
    datasetarff.class_is_last()  #Establecemos que el ultimo atributo corresponde a la clase.
    print(" -------> ARBOL PODADO <-------\n")
    evaluation.test_model(clasificador, datasetarff) #Evaluamos el dataset de prueba con el arbol podado...
    print(evaluation.matrix())  #... y mostramos su matriz de confusion.
    print(" -------> ARBOL COMPLETO <-------\n")
    evaluation2.test_model(clasificador2, datasetarff) #Evaluamos el dataset de prueba con el arbol completo...
    print(evaluation2.matrix())  #... y mostramos su matriz de confusion.
    print("  ----- Modelo nuevo -----")
    kNeighbors(kvecinos)  #Finalmente ejecutamos KNN con ya con los atributos del arbol podado.


leerT()  #Leemos el conjunto de Entrenamiento
leerP()  #Leemos el conjunto de Prueba
contarNomi()  #Hacemos los conteos para el calculo HVDM nominal.
desviacion()  #Calculamos las desviaciones estandar de atributos numericos.
for i in atributosEntrenamiento[:-1]:  #Declaramos a todos los atirbutos en luz verde para realizar calculos HVDM.
    habilitados.append(1)
jvm.start(packages=True)  #Encendemos una maquina virtual Java para el funcionamiento de weka.

while True:
    print("\n ---Main---\n1. KNN\n2. Arbol y KNN\n0. Salir")
    op = input()
    op = int(op)
    if op == 1:
        while True:  #Ciclo que termina hasta que el usuario ingrese el numero de k vecinos y cumpla las normas.
            print("Ingresa el numero de vecinos: ")
            tam = input()
            try:
                tam = int(tam)  #Verificamos que lo que haya ingresado sea un numero, caso contrario entra al except.
                if(tam%2 != 0):  #Si efectivamente es numero, revisamos que el numero sea impar.
                    tam = int(tam)
                    break  #Rompemos el ciclo si cumple las dos reglas anteriores.
                else:
                    print("--Ingresa un numero impar--")
            except:
                print("! Por favor ingresa un valor tipo 'int' !")
        habilitados.clear()  #Limpieza para volver a activar todos los atributos (por si se ha usado arbol).
        for i in atributosEntrenamiento[:-1]:
            habilitados.append(1)
        kNeighbors(tam)  #Algoritmo knn.
    if op == 2:
        while True:  #Mantenemos este while hasta que el usuario ingrese el numero de vecinos.
            print("Ingresa el numero de vecinos: ")
            tam = input()
            try:
                tam = int(tam)  #Verificamos que lo que haya ingresado sea un numero, caso contrario entra al except.
                if(tam%2 != 0):  #Si efectivamente es numero, revisamos que el numero sea impar.
                    tam = int(tam)
                    break  #Rompemos el ciclo si cumple las dos reglas anteriores.
                else:
                    print("--Ingresa un numero impar--")
            except:
                print("! Por favor ingresa un valor tipo 'int' !")
        arbol(tam)  #Creacion del arbol y algoritmo knn usando los atributos del arbol.
    if op == 0:
        print("Nos vemos!")
        jvm.stop()  #Al terminar la ejecucion, cerramos dicha maquina virtual.
        break
    if op < 0 or op > 2:
        print("!Opcion invalida!")