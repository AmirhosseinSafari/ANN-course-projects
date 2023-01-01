clc, clearvars, close all

% Reading the iris file's data 
file = fopen('iris.data');
rawdata = textscan(file,'%f %f %f %f %s', 150, 'Delimiter',',');

% Separating target data column
data = cell2mat(rawdata(:,1:4));
target = rawdata{1,5};

% labeling each category
[row,col] = size(target);
target_labels = zeros(row, 1);

for i = 1:row
    temp = target(i);
    if strcmp(temp,'Iris-setosa') == 1
        label = -1;
    elseif strcmp(temp,'Iris-versicolor') == 1
        label = 0;
    else
        label = 1;
    end

    target_labels(i, 1) = label;

end

% Combining training data and their labels
data = [data target_labels];

figure(1);
plot(data(:,3),data(:,4),'o','MarkerSize',5)
title ' Iris Data';
xlabel 'Petal length in cm'; 
ylabel 'Petal width in cm';

% divide data to train and test data
% traning data = 80%, test data = 20%
cv = cvpartition(size(data,1),'HoldOut',0.2);
idx = cv.test;

% Separating training and test data
training_data= data(~idx,:);
test_data= data(idx,:);

% Kmeans Implementation
k = 3;
[idx,C] = kmeans(training_data,k);
% size(C)
% size(data)

% Defining a grid 
x1 = min(training_data(:,3)):0.01:max(training_data(:,3));
x2 = min(training_data(:,4)):0.01:max(training_data(:,4));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)];

idx2Region = kmeans(XGrid,3,'MaxIter',100);

figure(2);
gscatter(XGrid(:,1),XGrid(:,2),idx2Region, [0,1,1; 1,0,1; 1,1,0],'..');
hold on;
plot(training_data(:,3),training_data(:,4),'ko','MarkerSize',5);
title 'Iris Data';
xlabel 'Petal length in cm'; 
ylabel 'Petal width in cm'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

% Calculating spread by finding variance
max_distance = 0;
for i = 1:k-1
    my_dist(i) = sum((C(i,:) - C(i+1,:)).^2);
    if  my_dist(i) > max_distance
            max_distance = my_dist(i);
    end
end

spread = (max_distance/sqrt(2*k))*ones(k,1);


train_target = training_data(:,5);
dataTrain = training_data(:,1:4);
test_target = test_data(:,5);
dataTest = test_data(:,1:4);
C = C(:,1:4);

correct = zeros( length(test_target),1 );
for i = 1:length(test_target)
    if test_target(i) == 1
        correct( i ) = 3;
    elseif test_target(i) == 0
        correct( i ) = 2;
    else
        correct( i ) = 1;
    end
end

y_one_hot = zeros( size( train_target, 1 ), 3 );
for i = 1:length(train_target)
    if train_target(i) == 1
        y_one_hot( i, 3 ) = 1;
    elseif train_target(i) == 0
        y_one_hot( i, 2 ) = 1;
    else
        y_one_hot( i, 1 ) = 1;
    end
end

% Finding weights
[m,n] = size(dataTrain);
%params = rand(n,k);
goal = zeros(k,m);
for i=1:m
    sample = dataTrain(i,:);
    for j = 1:k
        goal(j,i) = exp(-(sample-C(j,:))*(sample-C(j,:))'/(2*spread(j)^2));
    end
end
params = pinv(goal)'* y_one_hot;

% Applying RBF to test set
for i = 1:length(dataTest)
    sample = dataTest(i,:);

    for j = 1:k
        goal_test(j,i) = exp(-(sample-C(j,:))*(sample-C(j,:))'/(2*spread(j)^2));
    end
end

% Prediction and evaluate the model
prediction = (goal_test'* params);
for i = 1:length(dataTest)
    prediction(i) = softmax(prediction(i));
    [val, col] = max(prediction(i,:));
    predict(i) = col;
end
acc = (sum(predict==correct')/length(correct))*100;

fprintf("acc = %.3f\n", acc)