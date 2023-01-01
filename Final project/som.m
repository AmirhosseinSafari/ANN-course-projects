clc, clearvars, close all;


% Reading content of iris data
file = fopen('iris.data');
textdata = textscan(file,'%f %f %f %f %s', 150, 'Delimiter',',');

% Defining data matrix
data = cell2mat(textdata(:,1:4));
[m,n] = size(data);

% Plotting data, only first two features
figure(1);
plot(data(:,1),data(:,2),'k*','MarkerSize',5) 
title ' Iris Data';
xlabel 'sepal length (cm)';
ylabel 'sepal Width  (cm)';

% ======================= SOM Structure ======================= %

% Determining the number of rows and columns in the som
som_row = 20;
som_col = 20;

% =========== initialization =========== %

% Number of epochs
epoch = 500;

% Winner neuron neighbour
width_initial = 8;

learning_rate = 1;

a_max = 0.9 ;
a_min = 0.1 ;
t_max = epoch;

t_width = epoch/log(width_initial);

% =========== initialization weights of matrix =========== %
weight = zeros(som_row, som_col, n);

for i=1:som_row	
	for j=1:som_col
		weight(i, j, :) = rand(1,n);
	end
end

% =========== Training the model =========== %
for iter=1:epoch
    
    % Calculating varience which in each iteration changes
    width_varience = (width_initial * exp(-iter / t_width))^2;
	learning_rate = (a_max - a_min)* ((t_max - iter)/(t_max - 1)) + a_min;
    
    % Finding distance matrix of all data from weight matrix
    distance = zeros(som_row, som_col);
    i = randi([1 m]);
    for row = 1:som_row
        for col = 1:som_col
            sub = data(i,:) - reshape(weight(row,col,:),1,n);
            distance(row,col) = sqrt(sub * sub');
        end
    end
    
    % Finding minimum and winner neuron
    [minm,ind] = min(distance(:));
	[row_winner,col_winner] = ind2sub(size(distance),ind);
    
    % =========== Updating weights =========== %
    % Finding new weights of winner neuron and it's neighbors 
    dist = zeros(som_row, som_col);
    
    for row = 1:som_row
       for col = 1:som_col
           if (row == row_winner) && (col == col_winner) % winner neuron
               dist(row,col) = 1;
           else
               distance = (row_winner - row)^2+(col_winner - col)^2;
               dist(row,col) = exp(-(distance^2)/((2*width_varience)^2)); % neighborhood function for other neurons
           end    
       end
    end
    
    for row = 1: som_row
       for col = 1:som_col
           
           % Reshapping the dimension of the current weight vector
           weight_vec = reshape(weight(row,col,:),1,n);
           
           % Updating the weight vector for each neuron
           weight(row,col,:) = weight_vec + learning_rate*dist(row,col)*(data(i,:)-weight_vec);
            
       end
    end

    % =========== visualization =========== %
    % Weight vector of neuron
    dot = zeros(som_row*som_col, n);
        
    matrix = zeros(som_row*som_col,1);
    matrix_old = zeros(som_row*som_col,1);
        
    ind = 1;  
    hold on;
    f1 = figure(1);
    set(f1,'name',strcat('Iteration #',num2str(iter)),'numbertitle','off');
    
    for r = 1:som_row
        for c = 1:som_col      
            dot(ind,:) = reshape(weight(r,c,:),1,n);
            ind = ind + 1;
        end
    end
    
    % Plot SOM
    for r = 1:som_row
        Row_1 = 1 + som_row*(r-1);
        Row_2 = r*som_row;
        Col_1 = som_row*som_col;
    
        matrix(2*r-1,1) = plot(dot(Row_1:Row_2,1),dot(Row_1:Row_2,2),'--ro','LineWidth',1,'MarkerEdgeColor','g','MarkerFaceColor','b','MarkerSize',3);
        matrix(2*r,1) = plot(dot(r:som_col:Col_1,1),dot(r:som_col:Col_1,2),'--ro','LineWidth',1,'MarkerEdgeColor','g','MarkerFaceColor','b','MarkerSize',3);
    
        matrix_old(2*r-1,1) = matrix(2*r-1,1);
        matrix_old(2*r,1) = matrix(2*r,1);
    
    end
    
    pause(0.1)
    % Deleting the previous iteration plot (but not the last epoch's plot)
    if iter~=epoch  
        for r = 1:som_row
            delete(matrix_old(2*r-1,1));
            delete(matrix_old(2*r,1));
            drawnow;
        end
    end
end




% Finding winner neuron for every sample for creating top map 
top_map = zeros(som_row, som_col);

for i=1:m

    distance = zeros(som_row, som_col);

    for row = 1:som_row
        for col = 1:som_col
		    sub = data(i,:) - reshape(weight(row,col,:),1,n);
			distance(row,col) = sqrt(sub * sub');
        end
    end
		
    [minm,ind] = min(distance(:));
	[row_winner,col_winner] = ind2sub(size(distance),ind);
		
    top_map(row_winner,col_winner) = i;
		
end

% Displaying top map
fprintf("Top map:\n\n");
disp(top_map);

figure(2)

heatmap(top_map);