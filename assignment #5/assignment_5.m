clc, clearvars, close all

% intializing weights, learning rate, error, error treshhold, beta epoc number and error list 
learning_rate = 0.1;
weight_1_0 = [rand(), rand(), rand()];
weight_1_1 = [rand(), rand(), rand()];
weight_2 = [rand(), rand(), rand()];

E = 0;
treshhold = 10e-4;
beta = 0.7;
epoc_number = 50000;
error_list = zeros(1, epoc_number);
tresh_epoc = 0;

points = [0 0 1 1; 0 1 0 1];
labels = [0 1 1 0];

%  normalaization
max_x = max(points(1,:));
max_y = max(points(2,:));

min_x = min(points(1,:));
min_y = min(points(2,:));

for i = 1:numel(points(1,:))
    points(1,i) = (points(1, i) - min_x)/(max_x - min_x);
end

for i = 1:numel(points(1,:))
    points(2,i) = (points(2, i) - min_y)/(max_y - min_y);
end
%

for epoc = 1:epoc_number
    epoc
    E = 0;
    for i = 1:length(points)
        out_1_0 = activation(points(1, i)*weight_1_0(1) + points(2, i)*weight_1_0(2) + weight_1_0(3) );
        out_1_1 = activation(points(1, i)*weight_1_1(1) + points(2, i)*weight_1_1(2) + weight_1_1(3) );

        out_2 = activation(out_1_0*weight_2(1) + out_1_1*weight_2(2) + weight_2(3) );
        target = labels(i);
        E = E + 1/2*((target - out_2)^2);

        fprintf("--------------------------\n")
        fprintf("x1 = %d    out2 =   %d \n", points(1, i), out_2)
        fprintf("x2 = %d    target = %d \n\n", points(2, i), target)

        delta_2 = (target - out_2) * (out_2) * (1 - out_2);
        
        weight_2(1) = weight_2(1) + learning_rate * delta_2 * out_1_0 * (1 - beta);
        weight_2(2) = weight_2(2) + learning_rate * delta_2 * out_1_1 * (1 - beta);
        weight_2(3) = weight_2(3) + learning_rate * delta_2 * 1 * (1 - beta);

        weight_1_0(1) = weight_1_0(1) + learning_rate * delta_2 * weight_2(1) * out_1_0 * (1 - out_1_0) * points(1, i) * (1 - beta); 
        weight_1_0(2) = weight_1_0(2) + learning_rate * delta_2 * weight_2(1) * out_1_0 * (1 - out_1_0) * points(2, i) * (1 - beta); 
        weight_1_0(3) = weight_1_0(3) + learning_rate * delta_2 * weight_2(1) * out_1_0 * (1 - out_1_0) * 1 * (1 - beta);

        weight_1_1(1) = weight_1_1(1) + learning_rate * delta_2 * weight_2(2) * out_1_1 * (1 - out_1_1) * points(1, i) * (1 - beta);
        weight_1_1(2) = weight_1_1(2) + learning_rate * delta_2 * weight_2(2) * out_1_1 * (1 - out_1_1) * points(2, i) * (1 - beta);
        weight_1_1(3) = weight_1_1(3) + learning_rate * delta_2 * weight_2(2) * out_1_1 * (1 - out_1_1) * 1 * (1 - beta);
        
    end

    % treshhold checking
    fprintf("Error = %f \n", E);
    error_list(epoc) = E;

    if E < treshhold
        tresh_epoc = epoc;
        break
    end
        
end

if tresh_epoc == 0
    tresh_epoc = epoc_number;
end

%  ploting points and lines
figure(1)
plotpv(points, labels)
hold on
grid on

x = linspace(-4,5,50);
y1 =-(weight_1_0(1)/weight_1_0(2))*x + weight_1_0(3)/weight_1_0(2) + 1.5;
y2 =-(weight_1_1(1)/weight_1_1(2))*x + weight_1_1(3)/weight_1_1(2) + 1.5;
plot(x,y1, 'r-', LineWidth=1.5);
plot(x,y2, 'r-', LineWidth=1.5);
hold on

% ploting error per epoc
figure(2)
x = 1:1:tresh_epoc;
error_list = error_list(1:tresh_epoc);
plot(x, error_list)
grid on

% ploting 3D inputs and final outputs
outputs = zeros(1, 4);
for i = 1:length(points)
    out_1_0 = activation(points(1, i)*weight_1_0(1) + points(2, i)*weight_1_0(2) + weight_1_0(3) );
    out_1_1 = activation(points(1, i)*weight_1_1(1) + points(2, i)*weight_1_1(2) + weight_1_1(3) );
    out_2 = activation(out_1_0*weight_2(1) + out_1_1*weight_2(2) + weight_2(3) );
    outputs(i) = out_2;
end

figure(3)
plot3(points(1,:), points(2,:), outputs, "o", 'Color','r', 'MarkerSize',10)
grid on

% sigmoid activation function
function out = activation(wx)
    out = tansig(wx);
end

