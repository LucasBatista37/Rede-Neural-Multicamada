#importa a biblioteca numpy para função de arrays e matemáticas
import numpy as np

#Função sigmoid que calcula uma saída de saída onde o intervalo é de 0 a 1
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#Função sigmoid que calcula a derivada 
def sigmoidDerivada(sig):
    return sig * (1 - sig)

#Vetores onde serão realizados os calculos e pesos são inicializados aleatoriamente
entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([[0], [1], [1], [0]])
pesos0 = 2*np.random.random((2,3)) - 1
pesos1 = 2*np.random.random(3,1) - 1

epocas = 1000000
taxaAprendizagem = 0.6
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    #Calculo da soma da Sinapse 0 onde é feita a multiplicação da camada de entrada pelos pesos0 depois a soma de todas Sinapses
    somaSinapse0 = np.dot(camadaEntrada, pesos0)

    #Função de ativação sigmoid da camada oculta  
    camadaOculta = sigmoid(somaSinapse0)
    
    #Calculo da soma da Sinapse 1 onde é feita a multiplicação da camada de entrada pelos pesos1 depois a soma de todas Sinapses
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    #Função de ativação sigmoid da camada de saída  
    camadaSaida = sigmoid(somaSinapse1)

    #Calcula o erro da camada de saída para fazer o backpropagation
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    
    #Calculo da deriva de saída para fazer o delta
    derivadaSaida = sigmoidDerivada(camadaSaida)
    #Calculo do delta de saída 
    deltaSaida = erroCamadaSaida * derivadaSaida

    #tranposição de pesos e delta da camada oculta
    pesos1Trasposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Trasposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    #tranposição da camada oculta
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    #transposição da camada de entrada
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
print(somaSinapse0)
print(camadaOculta)