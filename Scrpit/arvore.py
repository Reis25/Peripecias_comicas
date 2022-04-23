import sys 

class NodoArvore: 
    def __init__(self, chave = None, esquerda = None, direita = None):
        self.chave = chave
        self.esquerda = esquerda
        self.direita = direita
    
    def __repr__(self):
        return '%s <- %s -> %s' %(self.esq and self.esq.chave,
                                 self.chave,
                                 self.dir and self.dir.chave)


def em_ordem(raiz): 
    if not raiz:
        return
 
    # Visita filho da esquerda.
    em_ordem(raiz.esquerda)

    # Visita nodo corrente.
    print(raiz.chave),
 
    # Visita filho da direita.
    em_ordem(raiz.direita)
        
def insere(raiz, nodo):
    """Insere um nodo em uma árvore binária de pesquisa."""
    # Nodo deve ser inserido na raiz.
    if raiz is None:
        raiz = nodo

    # Nodo deve ser inserido na subárvore direita.
    elif raiz.chave < nodo.chave:
        if raiz.direita is None:
            raiz.direita = nodo
        else:
            insere(raiz.direita, nodo)

    # Nodo deve ser inserido na subárvore esquerda.
    else:
        if raiz.esquerda is None:
            raiz.esquerda = nodo
        else:
            insere(raiz.esquerda, nodo)

def busca(raiz, chave):
    """Procura por uma chave em uma árvore binária de pesquisa."""
    # Trata o caso em que a chave procurada não está presente.
    if raiz is None:
        return None
    
    # A chave procurada está na raiz da árvore.
    if raiz.chave == chave:
        return raiz
 
    # A chave procurada é maior que a da raiz.
    if raiz.chave < chave:
        return busca(raiz.direita, chave)
   
    # A chave procurada é menor que a da raiz.
    return busca(raiz.esquerda, chave)

def menor_chave(raiz):  #busca o menor valor da arvore
    nodo = raiz
    while raiz.esquerda != None:
        nodo = nodo.esquerda
    return nodo.chave 
        

if __name__ == '__main__':

    raiz = NodoArvore(40)
    
    raiz = NodoArvore(40)

    raiz.esquerda = NodoArvore(20)
    raiz.direita  = NodoArvore(60)

    raiz.direita.esquerda  = NodoArvore(50)
    raiz.direita.direita   = NodoArvore(70)
    raiz.esquerda.esquerda = NodoArvore(10)
    raiz.esquerda.direita  = NodoArvore(30)

    em_ordem(raiz)
    
    
    for chave in [20, 60, 50, 70, 10, 30]:
        nodo = NodoArvore(chave)
        insere(raiz, nodo)
            
    #Imprime o caminhamento em ordem da árvore.
  

    