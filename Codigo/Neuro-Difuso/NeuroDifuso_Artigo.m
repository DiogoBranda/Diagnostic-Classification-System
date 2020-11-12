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
TesteInput = readmatrix("Dados\Artigo\testInput.txt"); 
TesteOutput = readmatrix("Dados\Artigo\testOutput.txt");
Treino= readmatrix("Dados\Artigo\treino.txt");


%% Funcao auxiliar

Classificar =  @(y) (y>0.5);

%% Calcular tamanho dos vetores de dados
tamanho_data_treino = length(Treino); %Tamanho dos dados de treino
tamanho_data_teste = length(TesteInput); %Tamanho dos dados de teste


%% Normalizar entradas
TesteInput(:,1) = ((TesteInput(:,1) - min( TesteInput(:,1)))/(max(TesteInput(:,1))-min( TesteInput(:,1))))*2-1;
TesteInput(:,2) = ((TesteInput(:,2) - min( TesteInput(:,2)))/(max(TesteInput(:,2))-min( TesteInput(:,2))))*2-1;
TesteInput(:,3) = ((TesteInput(:,3) - min( TesteInput(:,3)))/(max(TesteInput(:,3))-min( TesteInput(:,3))))*2-1;

Treino(:,1) = ((Treino(:,1) - min( Treino(:,1)))/(max(Treino(:,1))-min( Treino(:,1))))*2-1;
Treino(:,2) = ((Treino(:,2) - min( Treino(:,2)))/(max(Treino(:,2))-min( Treino(:,2))))*2-1;
Treino(:,3) = ((Treino(:,3) - min( Treino(:,3)))/(max(Treino(:,3))-min( Treino(:,3))))*2-1;
%% Valores para o anfis
iteracoes = 10000;% Número de iteracoes maximas no treino
taxa_de_aprendizagem = 0.008; % taxa de aprendizagem
Nmemberships = 3;% Número de memberships por input

%% DEFINIÇÃO DA ANFIS
[fis,error,stepsize] = anfis([Treino(:,1:3) Treino(:,4)],Nmemberships,[iteracoes 0 taxa_de_aprendizagem 0.9 1.1],[1 0 0 1]);
%Anfis vai ser usada para devolver a FIS (Fuzzy inference system) isto sera
%usado para calcular as saidas dos inputs inseridos, para isso iremos usar
%a evalfis que vai avaliar a fis a uma dada entrada e retorna a saida esperada
%% 
Output_Rede_Treino = evalfis(fis,Treino(:,1:3));
Output_Rede_Treino_norm = Classificar(Output_Rede_Treino);
figure(1)
hold on
scatter(1:tamanho_data_treino,  Treino(:,4))
scatter(1:tamanho_data_treino, Output_Rede_Treino_norm, '.')
grid on
title('Dados de treino')
xlabel('Indice de input')
ylabel('Output')

num_erros_treino = sum( Output_Rede_Treino_norm ~= Treino(:,4) );

%% Calcular outputs finais do teste usando a evalfis
Output_rede_teste     = evalfis(TesteInput, fis);
Output_rede_teste_norm = Classificar(Output_rede_teste);

%% Visualizar os dados de teste
figure(2)
hold on
scatter(1:tamanho_data_teste,  TesteOutput)
scatter(1:tamanho_data_teste, Output_rede_teste_norm, '.')
grid on
title('Dados de teste')
xlabel('Indice de input')
ylabel('Output')
num_erros_teste = sum( Output_rede_teste_norm ~=TesteOutput) ;
%% erros
num_erros_treino
num_erros_teste
