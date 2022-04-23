

import cv2
import matplotlib.pyplot as plt

#%% DETECCAO DE UMA FACE
image_path = 'kobe_teste.jpg'
cascade_pah = 'haarcascade_frontalface_default.xml'

clf = cv2.CascadeClassifier(cascade_pah) # CRIA CLASSIFICADOR

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = clf.detectMultiScale(gray, 1.2, 10)

for (x, y, w, h) in faces:  # PERCORRER FACES
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2) # DESENHAR UM RETANGULO

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
cv2.waitkey(0)
cv2.destroyAllWindows()
#%% DETECCAO DE VARIAS FACES

# Define o caminho para o classificador, neste caso Haar Cascade
caminho_classificador = 'haarcascade_frontalface_default.xml'

# Define o caminho para a imagem
caminho_imagem = "imagem.jpg"
# Caso a image e/ou o classificador nao estejam na mesma pasta do arquivo 
# fonte eh preciso informar o caminho completo "C:\\Users\\..."

# Cria o classificador
classificador = cv2.CascadeClassifier(caminho_classificador)

# Le a imagem
imagem = cv2.imread(caminho_imagem)

# Converte a imagem para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecta as faces na imagem
faces = classificador.detectMultiScale(
    cinza,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
# Desenha um retangulo para cada face encontrada
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(imagem)

#%% Uma face com olhos

# Define o caminho para o classificador, neste caso Haar Cascade
caminho_classificador = '/home/alunoic/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'

caminho_classificador_olhos = '/home/alunoic/Downloads/opencv-master/data/haarcascades/haarcascade_eye.xml'
# Define o caminho para a imagem
caminho_imagem = "kobe_teste.png"
# Caso a image e/ou o classificador nao estejam na mesma pasta do arquivo 
# fonte eh preciso informar o caminho completo "C:\\Users\\..."

# Cria o classificador
classificador = cv2.CascadeClassifier(caminho_classificador)

classificador2 = cv2.CascadeClassifier(caminho_classificador_olhos)

# Le a imagem
imagem = cv2.imread(caminho_imagem)

# Converte a imagem para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecta as faces na imagem
faces = classificador.detectMultiScale(
    cinza,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
# Desenha um retangulo para cada face encontrada
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(imagem)
    eyes = classificador2.detectMultiScale(cinza)
    for (ex,ey,ew,eh) in eyes:
        eye_detect = cv2.rectangle(imagem,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
        plt.imshow(eye_detect)

