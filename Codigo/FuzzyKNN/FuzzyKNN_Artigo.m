%##############################################
%                  FUZZY KNN                  #
%Desenvolvido por :                           #
%Diogo Silva u201809213                       #
%Fabio Morais u201504257                      #
%                                             #
%##############################################

%% Limpar consola
close all;
clear all;
clc;

%% Ler dados
TesteInput = readmatrix("Dados\Artigo\testInput.txt"); 
TesteOutput = readmatrix("Dados\Artigo\testOutput.txt");
Treino= readmatrix("Dados\Artigo\treino.txt");

%% Escolha de variaveis padrao
indeterminados=0;
erradas=0;
corretas=0;
K=3;

%% Calcular tamanho dos vetores de dados
tamanho_data_treino = length(Treino); %Tamanho dos dados de treino
tamanho_data_teste = length(TesteInput); %Tamanho dos dados de teste

%% Criar dados de output
Dados_Treino_Output_norm = Treino(:,4);


%% Algoritmo

for i=1:tamanho_data_teste % Para cada ponto em que nao conheça o seu valor de output vai tentar descobir o seu valor atravez dos seu vizinhos
    
    distancia = zeros(1,tamanho_data_treino);%vetor que contem as distancias entre os pontos conhecidos e o ponto desconhecido
    
    % 1 calcular distancias
    for j=1:tamanho_data_treino % Hip^2=cat1^2+cat2^2
        distancia(1,j) = sqrt((TesteInput(i,1)- Treino(j,1))^2 +(TesteInput(i,2)- Treino(j,2))^2+(TesteInput(i,3)- Treino(j,3))^2);
    end
    
    menorDist = zeros(1,K); %guarda as menores distancias
    indexMinDist = zeros(1,K);  %guarda os index das distancias
    for j=1:K % seleciona as K menores distancias
        [menorDist(1,j),indexMinDist(1,j)] = min(distancia);%Vai busacar a menor distancia e devolve o numero e a posicao no vetor
        distancia(1,indexMinDist(1,j)) = 10000000000;%Para nao voltar a ir buscar a mesma distancia metemos essa como "infinita"
    end
    
    media = mean(menorDist); %Calcular a media das menores distancias para calcualar o sigma
    sumatorio = 0; %Para calcular a primeira parte do sigma (formula 6 do relatorio)
    
    % 3 Calcular Sigma
    for j=1:K
        sumatorio= sumatorio + (menorDist(1,j)  - media)^2; %Sumatorio (d_i - d_medio)^2
    end
    sigma=sqrt( sumatorio/K); %sigma
    
    Di=[];%Numero difuso distancia crescente
    D0=[];%Numero difuso com [sigma, 0] é o denominador da formula (5)
    for j=1:K %criar numerador e denuminador
        Di=[Di;gaussmf(0:0.0001:sqrt(3),[sigma menorDist(1,j)])]; %numerador eq (5)
        D0=[D0;gaussmf(0:0.0001:sqrt(3),[sigma 0])];% denominador da formula (5)
    end
    Di=Di./(D0+1e-100);%calculo final do numero difuso eq(5) adicionamos um numero muito pequeno para nao dar erro a dividir por zero
    
    A = zeros(K,2);% Matriz A coluna 1 contem a classe "Nao soldado ->0" a coluna dois tem "Soldado ->1"
    
    for j=1:K
        %Fuzificar matriz A
        if(Dados_Treino_Output_norm(indexMinDist(1,j))==1) %se tiver soldado(=1) entao adiciona na segundo coluna
            A(j,2)=Di(j,1);% Vai buscar os valores á primeira coluna da matriz Di
        else
            A(j,1)=Di(j,1);% Senao adiciona na primeira
        end
    end
    W = ones(1,K);
    R=eye(K);%Matriz identidade
    resultado = W * R *A; %equaçao (12) Agora é a matriz A fuzificada e nao a R
    [Valor,index]= max(resultado);%equaçao (12)
    if resultado(1,1)==resultado(1,2)%Se as duas colunas tiverem 1 o algoritmo escolheu as duas classes como valor final logo uma indeterminaçao
        index=0;
        indeterminados=indeterminados+1;
    elseif(TesteOutput(i)==index-1)
        corretas=corretas+1;
    else
        erradas = erradas+1;
    end
end
corretas
erradas
indeterminados
