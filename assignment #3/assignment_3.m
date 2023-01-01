clc, clearvars, close all

learning_rate = 0.2;
weight = [rand, rand]; 
teta = rand;

% 10 points with labels as 3rd argument
points = {[1,4,1] ,[2,4,1], [1,3,1], [1,5,1], [2,3,1], [1,2,0], [1,1,0], [0,2,0], [2,1,0], [2,2,0]};

% drawing points
figure(1)

for c = 1:length(points)
    if (points{c}(3) == 1)
        scatter(points{c}(1), points{c}(2), 30, "green", "filled")
    else
        scatter(points{c}(1), points{c}(2), 30, "red", "filled")
    end
    hold on
    grid on
end

title('Perceptron Learning')

% Training
epoc_number = 300;
for epoc = 1:epoc_number
    epoc

    for i = 1:length(points)
        out = activation( points{i}(1)*weight(1) + points{i}(2)*weight(2) + (1)*(-1)*(teta) );
        amount_of_point = sqrt(points{i}(1)^2 + points{i}(2)^2);
        target = points{i}(3);
        weight(1) = weight(1) + (target - out)*(out)*(1-out)*(learning_rate * points{i}(1)/amount_of_point);
        weight(2) = weight(2) + (target - out)*(out)*(1-out)*(learning_rate * points{i}(2)/amount_of_point);
        
        teta = teta + (-1)*(target - out)*learning_rate;
    end
    
    % ploting the result line
    x =linspace(-4,5,50);
    y1 =(-weight(1)/weight(2))*x+teta/weight(2);
    plot(x,y1, 'y-', LineWidth=1.2)
    hold on

    % checking if all inputs seprated good
    y2 =@(x) (-weight(1)/weight(2))*x+teta/weight(2);
    all_data_ok = true;
    for j = 1:length(points)
        y2_out = y2(points{j}(1));
        if ~(( points{j}(2) - y2_out > 0 && points{j}(3) == 1 ) || ( points{j}(2) - y2_out < 0 && points{j}(3) == 0 ))
            all_data_ok = false;
            all_data_ok
            break
        end
    end
    
    if (all_data_ok == true)
        plot(x,y1, 'r-', LineWidth=1.5);
        break
    end   
end

% sigmoid activation function
function out = activation(wx)
    y_sigmoid = 1/(1+exp(-wx));
    out = y_sigmoid;
end