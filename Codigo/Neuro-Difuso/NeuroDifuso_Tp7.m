%##############################################
%            Sistema Neuro-Difuso             #
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

%% Funcao auxiliar
Classificar =  @(y) (y>0)*2-1;%% Classificar a saida

%% Calcular tamanho dos vetores de dados

tamanho_data_treino = find(ismember(Input,[0 0 0],'rows'))-1; %Encontrar [0 0 0] para calcular o tamanho dos dados de treino
tamanho_data_teste = length(Input)-tamanho_data_treino-1; %Tamanho dos dados de teste
%% Criar vetores de dados de input
Dados_Treino = Input(1:tamanho_data_treino,:);
Dados_Teste = Input(tamanho_data_treino+2:tamanho_data_treino+tamanho_data_teste+1,1:2);
%% Valores para o anfis
iteracoes = 1000;% Número de iteracoes maximas no treino
taxa_de_aprendizagem = 0.01; % taxa de aprendizagem
Nmemberships = 3;% Número de memberships por input

%% Normalizar entrada
Dados_Treino(:,1) = ((Dados_Treino(:,1) - min( Dados_Treino(:,1)))/(max(Dados_Treino(:,1))-min( Dados_Treino(:,1))))*2-1;
Dados_Treino(:,2) = ((Dados_Treino(:,2) - min( Dados_Treino(:,2)))/(max(Dados_Treino(:,2))-min( Dados_Treino(:,2))))*2-1;

Dados_Teste(:,1) = ((Dados_Teste(:,1) - min( Dados_Teste(:,1)))/(max(Dados_Teste(:,1))-min( Dados_Teste(:,1))))*2-1;
Dados_Teste(:,2) = ((Dados_Teste(:,2) - min( Dados_Teste(:,2)))/(max(Dados_Teste(:,2))-min( Dados_Teste(:,2))))*2-1;
%% DEFINIÇÃO DA ANFIS
[fis,error,stepsize] = anfis([Dados_Treino(:,1:2) Dados_Treino(:,3)],Nmemberships,[iteracoes 0 taxa_de_aprendizagem 0.9 1.1],[1 0 0 1]);
%Anfis vai ser usada para devolver a FIS (Fuzzy inference system) isto sera
%usado para calcular as saidas dos inputs inseridos, para isso iremos usar
%a evalfis que vai avaliar a fis a uma dada entrada e retorna a saida esperada

%% dados de treino 
Output_Rede_Treino = evalfis(fis,Dados_Treino(:,1:2));
Output_Rede_Treino_norm = Classificar(Output_Rede_Treino);
figure(1)
hold on
scatter(1:tamanho_data_treino,Dados_Treino(:,3))
scatter(1:tamanho_data_treino, Output_Rede_Treino_norm, '.')
grid on
title('Dados de treino')
xlabel('Indice de input')
ylabel('Output')

num_erros_treino = sum( Output_Rede_Treino_norm ~= Dados_Treino(:,3) );


%% Calcular outputs finais do teste usando a evalfis
Output_rede_teste     = evalfis(fis,Dados_Teste);
Output_rede_teste_norm = Classificar(Output_rede_teste);

%% Visualizar os dados de teste
  figure(2)
  hold on
  scatter(1:tamanho_data_teste,  Output)
  scatter(1:tamanho_data_teste, Output_rede_teste_norm, '.')
  grid on
  title('Dados de teste')
  xlabel('Indice de input')
  ylabel('Output')
  num_erros_teste = sum( Output_rede_teste_norm ~=Output) ;

%% erros
num_erros_treino
num_erros_teste
