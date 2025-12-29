clear; clc;
loop = 20;
SNR = zeros(loop,1);
epsilon = zeros(loop,1);
rel_err = zeros(loop,1);
t = 5; 
N = 1000; % Signal dimension
K = 5;   % Sparsity
step = 100;
M = 0;   % sampling number
SNRM = zeros(t*N/step,1);
EPSILONO = zeros(t*N/step,1);
Rel_Err = zeros(t*N/step,1);
ROW = zeros(t*N/step,1);

for i = 1:(t*N/step)
    M = M + step;  
    for j = 1:loop
        % 1. Generating a K-sparsity signal
        x0 = zeros(N,1);
        rp = randperm(N);
        x0(rp(1:K)) = randn(K,1); 
        x0 = x0/norm(x0);
        
        % 2. Call modified BIHT with ARIT thresholding
[xhat, BIHT_nbiter, BIHT_l2_err, BIHT_Hamming_err] = BIHT_with_ARIT_momentum(x0, M, N, K, 1000);
        
        % 3. Calculate performance metrics
        SNR(j) = 20*log10(norm(x0)/norm(xhat-x0));
        epsilon(j) = acos(dot(x0,xhat))/pi;
        rel_err(j) = BIHT_l2_err;
        
        fprintf(['M = ' num2str(M) ', loop = ' num2str(j) '\n']);
    end
    
    % 4. Calculate average performance
    SNRM(i) = sum(SNR)/loop;
    EPSILONO(i) = (sum(epsilon)/loop)^(-1);
    Rel_Err(i) = sum(rel_err)/loop;
    ROW(i) = M/N;
end

% 5. Save results
save('BIHT_ARIT_results.mat', 'SNRM', 'EPSILONO', 'ROW');

% 6. Plot results
figure;
subplot(2,2,1);
plot(ROW, SNRM, '-ro', 'MarkerFaceColor', 'g', 'LineWidth', 1.5);
axis([0 2 -inf inf]);
xlabel('M/N');
ylabel('SNR (dB)');
title('BIHT with ARIT Thresholding - SNR');
set(gca, 'linewidth', 1.5, 'fontsize', 13);
grid on;

subplot(2,2,2);
plot(ROW, EPSILONO, '-bo', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
xlabel('M/N');
ylabel('(Angular Error)^{-1}');
title('BIHT with ARIT Thresholding - Angular Error');
set(gca, 'linewidth', 1.5, 'fontsize', 13);
grid on;

subplot(2,2,3);
plot(ROW, Rel_Err, '-bo', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
xlabel('M/N');
ylabel('Relative Error');
title('BIHT with ARIT Thresholding - Angular Error');
set(gca, 'linewidth', 1.5, 'fontsize', 13);
grid on;

