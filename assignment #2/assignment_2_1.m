clc, clearvars, close all

learning_rate = 0.3;
weight = [rand, rand];
teta = rand;
points = [0 0 1 1; 0 1 0 1];
labels = [1 0 0 0];
plotpv(points,labels)
hold on
grid on

epoc_number = 200;
for epoc = 1:epoc_number
    epoc

    for i = 1:length(points)
        out = activation( points(1, i)*weight(1) + points(2, i)*weight(2) + (1)*(-1)*(teta) );
        target = labels(i);
        weight(1) = weight(1) + (target - out )*(learning_rate * points(1, i));
        weight(2) = weight(2) + (target - out )*(learning_rate * points(2, i));
        teta = teta + (-1)*(target - out)*learning_rate;
    end
    out
    % ploting the result line
    x =linspace(-4,5,50);
    y1 =(-weight(1)/weight(2))*x+teta/weight(2);
    plot(x,y1, 'y-', LineWidth=1.2);
    
    y2 =@(x) (-weight(1)/weight(2))*x+teta/weight(2);
    all_data_ok = true;
    for j = 1:length(points)
        y2_out = y2(points(1, j));
        if ~(( points(2, j) - y2_out > 0 && labels(j) == 0 ) || ( points(2, j) - y2_out < 0 && labels(j) == 1 ))
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

% sign activation function
function out = activation(wx)
    if (wx >= 0)
        out = 1;
    else
        out = 0;
    end
end
