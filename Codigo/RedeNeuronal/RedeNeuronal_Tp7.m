close all
clear all
clc
index_funcActivation=1; % escolhe fun�ao de ativa�ao
%%
switch index_funcActivation
  case 1  % FUN��O DE ATIVA��O TANGENTE HIPERB�LICA
    inMin = -1;
    inMax = +1;
    normaliza = @(x,min,max) ((x-min)/(max-min)*2-1);
    normalizaOut   = @(z) z;
    activationFunction = @(x) tanh(x); 
    activationFunction_derivative = @(x) (sech(x)).^2;    
    normalizaSaida = @(z) (z > 0)*2-1; % entre -1 e 1
  case 2  % FUN��O DE ATIVA��O SIGMOID 
    inMin =  0;
    inMax = +1;  
    normaliza = @(x,min,max) ((x-min)/(max-min));
    normalizaOut   = @(z) (z>0);
    activationFunction = @(x) 1./(1+exp(-x));
    activationFunction_derivative = @(x) exp(-x)./(exp(-x) + 1).^2;
    normalizaSaida = @(z) (z > 0.5); % entre 0 e 1
end

%%

inputFile = 'Dados/P2/testInput11B.txt';
outputFile = 'Dados/P2/testOutput11B.txt';
inputData = readmatrix(inputFile); % Leitura input
outputData = readmatrix(outputFile); % Leitura output

trainSize = find(ismember( inputData, [0 0 0], 'rows' )) - 1; % tamanho do treino
testSize = length(inputData) - find(ismember( inputData, [0 0 0], 'rows' )) ; %tamanho do teste

if length(outputData) ~= testSize
    disp('tamanho diferente');
    return;
end

trainData =  inputData( 1:trainSize , : );
testData =  inputData(  trainSize + 2 : trainSize + testSize + 1 , 1:2);


%% normaliza dados

trainDataNormalizado(:,1) = normaliza( trainData(:,1), min(trainData(:,1)), max(trainData(:,1)) ); % Normaliza��o dos dados de entrada para treino da rede
trainDataNormalizado(:,2) = normaliza( trainData(:,2), min(trainData(:,2)), max(trainData(:,2)) ); % Normaliza��o dos dados de entrada para treino da rede
trainDataNormalizado(:,3) = normalizaOut(trainData(:,3));

testeDataNormalizado(:,1) = normaliza( testData(:,1), min(trainData(:,1)), max(trainData(:,1)) ); % Normaliza��o dos dados de entrada para treino da rede
testeDataNormalizado(:,2) = normaliza( testData(:,2), min(trainData(:,2)), max(trainData(:,2)) ); % Normaliza��o dos dados de entrada para treino da rede


%% fun�oes
random = @(m,n) rand(m,n)/2;            % gera os pesos entre 0 a 0.5

sumNeuro = @(x, w, b) sum(x * w) + b;
saidaNeuro = @(x, w, b) activationFunction(sumNeuro(x,w,b));

%% Treinar
%exemplo de configura�ao 5:2
%       o \
%x1---/ o-o\
%       o   o--
%x2---\ o-o/
%       o/

% Configurar a rede
nosCamada1=5; % nos da camada 1
nosCamada2=3; % nos da camada 2

w0 = random(nosCamada1,2);
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

h1=zeros(nosCamada1,1); % hidden layer com 4 n�s
h2=zeros(nosCamada2,1); % hidden layer com 2 n�s

while erros > 0 && numIteracao < maxNumIteracao
    for i=1 : trainSize
        h1 = sum((w0.*trainDataNormalizado(i,1:2)).')  + b1;
        h2 = activationFunction(h1) * w1 + b2;
        uk =  sum(w2 .* activationFunction(h2)) + b3;
        z(i) = activationFunction(uk);
        
        % Erros
        ez= (trainDataNormalizado(i,3) - z(i)) * activationFunction_derivative(uk); % diferen�a entre o desejado e saida
        eh2=((w2*ez.*activationFunction_derivative(h2)).' );
        eh1=(w1*eh2.*activationFunction_derivative(h1.'));

        w0= w0+ eh1*trainDataNormalizado(i,1:2)*taxaAprendizagem;
        w1 = w1 + (eh2.*activationFunction(h1)*taxaAprendizagem).';
        w2 = w2 + ez*activationFunction(h2)*taxaAprendizagem ;
        
        b1 = b1 +  (eh1*taxaAprendizagem).'; % pertence ao h1
        b2 = b2 +  (eh2*taxaAprendizagem).'; % pertence ao h2
        b3 = b3 +  (ez*taxaAprendizagem).'; % pertence ao z

        
    end
    saidaZ = normalizaSaida(z);
    erros=0;
    for i=1 : trainSize
        if saidaZ(i) ~= trainDataNormalizado(i,3)
            erros= erros + 1;
        end
    end
    if mod(numIteracao, 100) == 0
        fprintf('[%d] - erros %d\n', numIteracao, erros);
    end
    
numIteracao = numIteracao + 1;
end


%% testar com os valores que foram dados de teste

erros=0;
for i=1 : testSize
     h1 = sum((w0.*testeDataNormalizado(i,1:2)).')  + b1;
     h2 = activationFunction(h1) * w1 + b2;
     uk =  sum(w2 .* activationFunction(h2)) + b3;
     z(i) = activationFunction(uk);
    saidaTest = normalizaSaida(z(i));
    saidaReal = normalizaOut(outputData(i));
    if saidaTest ~= saidaReal
        erros= erros+1;
        fprintf('[%i]-Saida teste: %d saida real: %d \n',i, saidaTest, saidaReal);

    end
end
fprintf('[%s]  numero de itera�ao= %d\t numero de erros: %d\t n� amostras: %d\t taxa de sucesso: %.1f%% \n',inputFile, numIteracao,erros, testSize,(testSize-erros)/testSize *100);

%%
red  = [];  % Classe -1
blue = [];  % Classe +1

% Pontos relativos ao treino da rede
for i=1:trainSize
  if trainData(i,3) == -1
    red  = [ red  ; trainDataNormalizado(i, 1:2) ];  % Adicionar valores da entrada quando o exemplo � da classe RED
  else
    blue = [ blue ; trainDataNormalizado(i, 1:2) ];  % Adicionar valores da entrada quando o exemplo � da classe BLUE
  end
end
for i=1:testSize
  if outputData(i,1) == -1
    red  = [ red  ; testeDataNormalizado(i, :) ];     % Adicionar valores da entrada quando os dados de teste s�o da classe RED
  else
    blue = [ blue ; testeDataNormalizado(i, :) ];     % Adicionar valores da entrada quando os dados de teste s�o da classe BLUE
  end
end

x = inMin:0.001:inMax;
figure(1)
hold on
xlim([ inMin inMax ])
ylim([ inMin inMax ])
scatter( red(:,1) , red(:,2) , 25, 'red' , 'filled')% Pontos pertencentes � classe -1
scatter( blue(:,1), blue(:,2), 25, 'blue', 'filled')% Pontos pertencentes � classe +1
scatter( testeDataNormalizado(:,1), testeDataNormalizado(:,2), 60, 'black') % Pontos de teste da rede neuronal

%plot(x, (-b(1,1)-w(1,1)*x)/w(1,2), 'LineWidth', 1, 'Color', [0 0 0]) % Linha de decis�o das classes
legend('Classe -1', 'Classe +1', 'Teste')
xlabel('Entrada X \rightarrow')
ylabel('Entrada Y \rightarrow')
grid on
