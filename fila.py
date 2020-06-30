import sys
import traceback


class fila:
    def __init__(self):
        self.__fila =[]
    
    def __len__(self):
        return len(self.__fila)
    
    def fila_vazia(self):
        return len(self.__fila) == 0
    
    def inseri(self, elemento):
        self.__fila.append(elemento)
    
    def cabeca(self):
        if(self.fila_vazia()):
            raise(Empty('Stack is empty'))
        else:
            return self.__fila[0]

    def busca(self, elemento):
        for i in range(0, len(self.__fila)):
            if(self.__fila[i] == elemento):
                print("Elemento contido na lista na posicao {%d}", i)
    
    def imprime_fila(self): 
        for i in range(0, len(self.__fila)):
            print(self.__fila[i])
            

if __name__ == '__main__':
    
    fila_reis = fila()
    
    print("inicio da fila: \n \n")
    
    fila_reis.imprime_fila()
    
    print("inserindo elementos na fila: 1 2 3 4 5 6 7")
    
    fila_reis.inseri(1)
    fila_reis.inseri(2)
    fila_reis.inseri(3)
    fila_reis.inseri(4)
    fila_reis.inseri(5)
    fila_reis.inseri(6)
    fila_reis.inseri(7)
    
    fila_reis.imprime_fila()
    
    print("removendo elementos da fila ")
    
    print(fila_reis.cabeca())
    
    fila_reis.imprime_fila()