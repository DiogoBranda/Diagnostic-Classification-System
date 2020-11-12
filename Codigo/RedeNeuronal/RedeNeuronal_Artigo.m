close all
clear all
clc
index_funcActivation=1; % escolhe funçao de ativaçao
%%
switch index_funcActivation
  case 1  % FUNÇÃO DE ATIVAÇÃO TANGENTE HIPERBÓLICA
    inMin = -1;
    inMax = +1;
    normaliza = @(x,min,max) ((x-min)/(max-min)*2-1);
    normalizaOut   = @(z) (z > 0)*2-1;
    activationFunction = @(x) tanh(x); 
    activationFunction_derivative = @(x) (sech(x)).^2;    
    normalizaSaida = @(z) (z > 0)*2-1; % entre -1 e 1
  case 2  % FUNÇÃO DE ATIVAÇÃO SIGMOID 
    inMin =  0;
    inMax = +1;  
    normaliza = @(x,min,max) ((x-min)/(max-min));
    normalizaOut   = @(z) (z>0);
    activationFunction = @(x) 1./(1+exp(-x));
    activationFunction_derivative = @(x) exp(-x)./(exp(-x) + 1).^2;
    normalizaSaida = @(z) (z > 0.5); % entre 0 e 1
end

%%

inputFile = 'Dados/artigo/testInput.txt';
outputFile = 'Dados/artigo/testOutput.txt';
trainFile = 'Dados/artigo/treino.txt';

testData = readmatrix(inputFile); % Leitura input
outputData = readmatrix(outputFile); % Leitura output
trainData = readmatrix(trainFile);

trainSize =length(trainData); % tamanho do treino
testSize = length(testData); %tamanho do teste

if length(outputData) ~= testSize
    disp('tamanho diferente');
    return;
end



% normaliza dados

trainDataNormalizado(:,1) = normaliza( trainData(:,1), min(trainData(:,1)), max(trainData(:,1)) ); % Normalização dos dados de entrada para treino da rede
trainDataNormalizado(:,2) = normaliza( trainData(:,2), min(trainData(:,2)), max(trainData(:,2)) ); % Normalização dos dados de entrada para treino da rede
trainDataNormalizado(:,3) = normaliza( trainData(:,3), min(trainData(:,3)), max(trainData(:,3)) ); % Normalização dos dados de entrada para treino da rede
trainDataNormalizado(:,4) = normalizaOut(trainData(:,4));

testeDataNormalizado(:,1) = normaliza( testData(:,1), min(trainData(:,1)), max(trainData(:,1)) ); % Normalização dos dados de entrada para treino da rede
testeDataNormalizado(:,2) = normaliza( testData(:,2), min(trainData(:,2)), max(trainData(:,2)) ); % Normalização dos dados de entrada para treino da rede
testeDataNormalizado(:,3) = normaliza( testData(:,3), min(trainData(:,3)), max(trainData(:,3)) ); % Normalização dos dados de entrada para treino da rede


%% funçoes
random = @(m,n) rand(m,n)/2;            % gera os pesos entre 0 a 0.5

sumNeuro = @(x, w, b) sum(x * w) + b;
saidaNeuro = @(x, w, b) activationFunction(sumNeuro(x,w,b));

%% Treinar
%exemplo de configuraçao 5:3
%       o \
%x1---/ o-o\
%       o-o-o--
%x2---\ o-o/
%       o/

% Configurar a rede
nosCamada1=5; % nos da camada 1
nosCamada2=3; % nos da camada 2

w0 = random(nosCamada1,3);
w1 = random(nosCamada1,nosCamada2); 
w2 = random(1,nosCamada2); 

b1 = random(1,nosCamada1); % para o H1, H2 e Z
b2 = random(1,nosCamada2);
b3 = random(1,1); 
eh1= zeros(nosCamada1,1);
eh2= zeros(nosCamada2,1);
ez= zeros(1,1);

erros = 1;
maxNumIteracao = 1000;
numIteracao = 0;
taxaAprendizagem = 0.2;
z = zeros(1,trainSize);

h1=zeros(nosCamada1,1); % hidden layer com 4 nós
h2=zeros(nosCamada2,1); % hidden layer com 2 nós

while erros > 0 && numIteracao < maxNumIteracao
    for i=1 : trainSize
        h1 = sum((w0.*trainDataNormalizado(i,1:3)).')  + b1;
        h2 = activationFunction(h1) * w1 + b2;
        uk =  sum(w2 .* activationFunction(h2)) + b3;
        z(i) = activationFunction(uk);
        
        % Erros
        ez= (trainDataNormalizado(i,4) - z(i)) * activationFunction_derivative(uk); % diferença entre o desejado e saida
        eh2=((w2*ez.*activationFunction_derivative(h2)).' );
        eh1=(w1*eh2.*activationFunction_derivative(h1.'));

        w0= w0+ eh1*trainDataNormalizado(i,1:3)*taxaAprendizagem;
        w1 = w1 + (eh2.*activationFunction(h1)*taxaAprendizagem).';
        w2 = w2 + ez*activationFunction(h2)*taxaAprendizagem ;
        
        b1 = b1 +  (eh1*taxaAprendizagem).'; % pertence ao h1
        b2 = b2 +  (eh2*taxaAprendizagem).'; % pertence ao h2
        b3 = b3 +  (ez*taxaAprendizagem).'; % pertence ao z
        
    end
    saidaZ = normalizaSaida(z);
    erros=0;
    for i=1 : trainSize
        if saidaZ(i) ~= trainDataNormalizado(i,4)
            erros= erros + 1;
        end
    end
    if mod(numIteracao, 100) == 0
        fprintf('[%d] - erros %d\n', numIteracao, erros);
    end
    
numIteracao = numIteracao + 1;
end


%% testar com os valores que foram dados de teste
fprintf("%d-sucesso treino : %f\n", erros, ((trainSize-erros)/trainSize) * 100);
erros=0;
for i=1 : testSize
     h1 = sum((w0.*testeDataNormalizado(i,1:3)).')  + b1;
     h2 = activationFunction(h1) * w1 + b2;
     uk =  sum(w2 .* activationFunction(h2)) + b3;
     z(i) = activationFunction(uk);
    saidaTest = normalizaSaida(z(i));
    saidaReal = normalizaOut(outputData(i));
    if saidaTest ~= saidaReal
        erros= erros+1;
        fprintf('[%i]-Saida teste: %d saida real: %d\t%f\t%f \n',i, saidaTest, saidaReal, z(i), outputData(i));
       
    end
end
fprintf('[%s]  numero de iteraçao= %d\t numero de erros: %d\t nº amostras: %d\t taxa de sucesso: %.1f%% \n',inputFile, numIteracao,erros, testSize,(testSize-erros)/testSize *100);

%
red  = [];  % Classe -1
blue = [];  % Classe +1

% Pontos relativos ao treino da rede
for i=1:trainSize
  if trainData(i,4) == 0
    red  = [ red  ; trainDataNormalizado(i, 1:3) ];  % Adicionar valores da entrada quando o exemplo é da classe RED
  else
    blue = [ blue ; trainDataNormalizado(i, 1:3) ];  % Adicionar valores da entrada quando o exemplo é da classe BLUE
  end
end
for i=1:testSize
  if outputData(i,1) == 0
    red  = [ red  ; testeDataNormalizado(i, :) ];     % Adicionar valores da entrada quando os dados de teste são da classe RED
  else
    blue = [ blue ; testeDataNormalizado(i, :) ];     % Adicionar valores da entrada quando os dados de teste são da classe BLUE
  end
end

figure(1)
hold on
xlim([ inMin inMax ])
ylim([ inMin inMax ])
zlim([ inMin inMax ])
scatter3( red(:,1) , red(:,2) ,red(:,3), 25, 'red' , 'filled')% Pontos pertencentes à classe -1
scatter3( blue(:,1), blue(:,2),blue(:,3), 25, 'blue', 'filled')% Pontos pertencentes à classe +1
scatter3( testeDataNormalizado(:,1), testeDataNormalizado(:,2), testeDataNormalizado(:,3), 60, 'black') % Pontos de teste da rede neuronal

%plot(x, (-b(1,1)-w(1,1)*x)/w(1,2), 'LineWidth', 1, 'Color', [0 0 0]) % Linha de decisão das classes
legend('NonWeld', 'Weld', 'Teste')
xlabel('Entrada U_i')
ylabel('Entrada V_i')
zlabel('Entrada W_i')
grid on
view(3)
