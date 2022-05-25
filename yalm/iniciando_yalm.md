 Lingagem de marcação / Linguage de srialização dos dados (serialização - desserialização dos dados); 

Serialização: é a técnica que permite converter objetos em bytes (colocando-os) em série e uma vez que eles são bytes,
eles podem ser salvos em disco ou enviados através de um stream (via HTTP, via socket, entre outros)

Possui os tipos de dados: escalares, arrays, lists; 

Geralmente é usada como arquivo de configuração ou armazenamento de dados; 

Tem como objetivo: 
- Ser portátil. 
- Integrar facilmente com outras liguagens; 
- Facil de implementar e usar; 
- fácil compreeensão humana; 

# Docker: 

* Pipeline básico para uso de docker numa aplicação. 

Buildando: 

~~~Shell
$ docker build -f HOTEL-API.Dockerfile -t reis25/hotel-api:v1 . 
~~~
 
reis25/HOTEL-API:v1 . ===> usuario/nomedaimagem:TagDeVersão

. (uso do ponto) menciona todos os arquivos do diretório. 

Verificando se a imagem foi criada corretamente: 

~~~Shell
$ docker run --name "hotel-api" -d -p 8080:80 reis25/hotel-api:v1
~~~

* Para verificar o funcionameno, acessa a porta especificada (no caso): http:localhost:8080

Push na Imagem para o DockerHub: 

~~~Shell
$ docker login --username=reis25
~~~

passaword: 
~~~Shell
$
~~~

Imagens locais: 

~~~Shell
$ docker image ls
~~~

Subindo para o repositório: 
~~~Shell
$ docker image push reis25/hotel-api:v1
~~~