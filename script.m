% LECTURA DE LOS DATOS
R=readtable("datos.csv");

% seleccionamos las columnas 
data=R{:,3:32};
target=R{:,2};

target=(target=="M")*2-1;

C=cvpartition(target,'HoldOut',0.25);
save("partition","C")

% CREACION DE MATRICES y NORMALIZACION DE LOS DATOS
% vectores de indices que indican que filas pertenecen al entrenamiento
train_id=C.training;
test_id=C.test;

% matrices de coeficientes de entrenamiento y test
Atrain=data(train_id, :);
Atest=data(test_id, :);

% terminos independientes de entrenamiento y test
btrain=target(train_id);
btest=target(test_id);


% normalizacion
mu=mean(Atrain);
sigma=std(Atrain);

Atrain=(Atrain-mu)./sigma;
Atest=(Atest-mu)./sigma;

% SCATTER PLOT

gscatter(Atrain(:,1), Atrain(:,8), btrain, ['r','b'], ['o','o'], 6);
xlabel('Radio medio (normalizado)');
ylabel('Puntos cóncavos medios (normalizado)');
title('Clasificación de tumores: Benigno vs Maligno');
legend('Maligno','Benigno', 'Location','northwest');
grid on;
saveas(gcf, 'grafico.png');

% REGRESION MULTILINEAL y CLASIFICACION

% regresion multilineal
Atrain_ext=[Atrain,ones(size(Atrain,1),1)]; % anadimos una columna de 1's, termino independiente.
Atest_ext=[Atest,ones(size(Atest,1),1)];
coef=Atrain_ext\btrain;


predictions_test=Atest_ext*coef;
predictions_train=Atrain_ext*coef;

% clasificacion y precision del test
classification_test=sign(predictions_test);
classification_train=sign(predictions_train);

correct_test=(classification_test==btest);
correct_train=(classification_train==btrain);

accuracy_test=sum(correct_test)/numel(correct_test)*100
accuracy_train=sum(correct_train)/numel(correct_train)*100

% ANALISIS DE CARACTERISTICAS

% obtenemos las columnas cuyos coef. en valor absoluto son mayores
[~, id_sorted]=sort(abs(coef(1:30)),'descend');
top5=id_sorted(1:5);

Atrain_top5=[Atrain(:,top5), ones(size(Atrain,1), 1)];
Atest_top5=[Atest(:, top5), ones(size(Atest,1), 1)];
coef_top5=Atrain_top5\btrain;

predictions_test_top5=Atest_top5*coef_top5;
predictions_train_top5=Atrain_top5*coef_top5;

% clasificacion y precision del test
classification_test_top5=sign(predictions_test_top5);
classification_train_top5=sign(predictions_train_top5);

correct_test_top5=(classification_test_top5==btest);
correct_train_top5=(classification_train_top5==btrain);

accuracy_test_top5=sum(correct_test_top5)/numel(correct_test_top5)*100
accuracy_train_top5=sum(correct_train_top5)/numel(correct_train_top5)*100