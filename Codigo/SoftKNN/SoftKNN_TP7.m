%##############################################
%                   SOFT KNN                  #
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
Input = readmatrix("Dados\P2\testInput11C.txt"); 
Output = readmatrix("Dados\P2\testOutput11C.txt");
%% Escolha de variaveis padrao
indeterminados=0;
erradas=0;
corretas=0; 
K=7;
%% funçoes auxiliares 
normalizar_out = @(y) (y+1)/2;
%% Calcular tamanho dos vetores de dados
tamanho_data_treino = find(ismember(Input,[0 0 0],'rows'))-1; %Encontrar [0 0 0] para calcular o tamanho dos dados de treino
tamanho_data_teste = length(Input)-tamanho_data_treino-1; %Tamanho dos dados de teste
%% Criar vetores de dados de input
Dados_Treino = Input(1:tamanho_data_treino,:);
Dados_Teste = Input(tamanho_data_treino+2:tamanho_data_treino+tamanho_data_teste+1,1:2);
%% Criar e normalizar dados de output para treino 
Dados_Treino_Output_norm = normalizar_out(Dados_Treino(:,3));
Output= normalizar_out( Output);
%% Algoritmo
for i=1:tamanho_data_teste % Para cada ponto em que nao conheça o seu valor de output vai tentar descobir o seu valor atravez dos seu vizinhos
   
    distancia = zeros(1,tamanho_data_treino);%vetor que contem as distancias entre os pontos conhecidos e o ponto desconhecido
    
    % 1 calcular distancias 
    for j=1:tamanho_data_treino % Hip^2=cat1^2+cat2^2
        distancia(1,j) = sqrt((Dados_Teste(i,1)- Dados_Treino(j,1))^2 +(Dados_Teste(i,2)- Dados_Treino(j,2))^2);
    end
    
    % 2 encontrar pontos mais proximos
    A = zeros(K,2);% Matriz A coluna 1 contem a classe "Nao soldado ->0" a coluna dois tem "Soldado ->1"
    menorDist = zeros(1,K); %guarda as menores distancias 
    for j=1:K % seleciona as K menores distancias
        [menorDist(1,j),indexMinDist] = min(distancia);%Vai busacar a menor distancia e devolve o numero e a posicao no vetor
        distancia(1,indexMinDist) = 10000000000;%Para nao voltar a ir buscar a mesma distancia metemos essa como "infinita"
        if Dados_Treino_Output_norm(indexMinDist,1)== 1 %se a classe se for 1 
            A(j,2)=1;%insere na coluna 2
        else % senao 
            A(j,1)=1;  %insere na coluna 2
        end
    end
   
    media = mean(menorDist); %Calcular a media das menores distancias para calcualar o sigma
    sumatorio = 0; %Para calcular a primeira parte do sigma (formula 6 do relatorio)
    
    % 3 Calcular Sigma
    for j=1:K 
        sumatorio= sumatorio + (menorDist(1,j)  - media)^2; %Sumatorio (d_i - d_medio)^2
    end
    sigma=sqrt( sumatorio/K); %sigma
    
    Di=[];%Numero difuso distancia crescente
    Dj=[];%Numero difuso distancia decrescente
    D0=[];%Numero difuso com [sigma, 0] é o denominador da formula (5)
    aux=K;%para conseguir calcular para o numero difuso Dj, no for faremos calcular de forma decrescente decrementando esta variavel
    % 4 calcular os numeros difusos para as distancias
    for j=1:K %criar numerador e denuminador
        Di=[Di;gaussmf(0:0.0001:sqrt(3),[sigma menorDist(1,j)])]; %numerador eq (5)
        Dj=[Dj;gaussmf(0:0.0001:sqrt(3),[sigma menorDist(1,aux)])]; %Para termos tambem a ordem decrescente para fazer a matriz aauxiliar alpha
        aux=aux-1;
        D0=[D0;gaussmf(0:0.0001:sqrt(3),[sigma 0])];% denominador da formula (5)
    end
    Di=Di./(D0+1e-100);%calculo final do numero difuso eq(5) adicionamos um numero muito pequeno para nao dar erro a dividir por zero
    Dj=Dj./(D0+1e-100);
    
    alpha=zeros(K,K);%matriz auxiliar iremos calcular de acordo com a formula (8)
    for j=1:K
        for l=1:K
            alpha(j,l)=max(min(Di(j,:),Dj(l,:))); % fizemos S como sendo max min
        end
    end
    teste_converge=zeros(K,K);%para verifica se a matriz alpha ja convergiu, ou seja se a alpha antiga é igual a nova entao terminamos
    R=zeros(K,K);%Matriz R
    while 1
        %normalizar linhas de alpha
        for j=1:K
            SumatorioLinhas = sum(alpha(j,:))+1e-100;% sumar numero muito pequeno para pervenir que tenhamos uma divisao por zero no passo seguinte
            for l=1:K
                alpha(j,l)=alpha(j,l)/SumatorioLinhas;%equaçao (9)
            end
        end
        %normalizar colunas de alpha
        for j=1:K
            Sumatoriocolunas = sum(alpha(:,j))+1e-100;% sumar numero muito pequeno para pervenir que tenhamos uma divisao por zero no passo seguinte
            for l=1:K
                alpha(l,j)=alpha(l,j)/Sumatoriocolunas;%equaçao (10)
            end
        end
        
        %verificar se convergiu se nao faz isto ate convergir
        if((round(teste_converge,10)==round(alpha,10))) %Quando as primeiras 10 casas decimais nao mudarem entao convergiu
            R= alpha;%se convergir a matriz R esta determinada e podemos sair do ciclo
            break;
        end
        teste_converge=alpha;%guarda o antigo valor de alpha para comparar
    end
    W = ones(1,K); 
    resultado = W * R *A; %equaçao (4)
    [Valor,index]= max(resultado);%equaçao (4)
    if resultado(1,1)==resultado(1,2)%Se as duas colunas tiverem 1 o algoritmo escolheu as duas classes como valor final logo uma indeterminaçao
        index=0;
        indeterminados=indeterminados+1;
    elseif(Output(i)==index-1)
        corretas=corretas+1;
    else
       erradas = erradas+1;
    end
end
corretas
erradas
indeterminados