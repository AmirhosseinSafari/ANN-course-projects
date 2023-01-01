clc, clearvars, close all
x = linspace(-5, 5)
y_sigmoid = 1./(1+exp(-x))

% Sigmoid function
figure(1)
subplot(1, 2, 1)
plot(x,y_sigmoid, '-g*')
xlabel('x'), ylabel('y'), title('Y vs X _ Sigmoid function')
grid on
legend('sigmoid')

% Gaussian function
sigma = std(x)
mu = mean(x)
y_gaussian = (1./(sqrt(2 * pi) * sigma)) * exp( (-1/2) * (((x - mu) ./ sigma) .^ 2)  )

subplot(1,2,2)
plot(x, y_gaussian, '--b+')
xlabel('x'), ylabel('y'), title('Y vs X _ Gaussian function')
grid on
legend('gaussian')