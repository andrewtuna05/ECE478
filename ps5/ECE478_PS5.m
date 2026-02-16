%Andrew Yuan ECE478 PS5
clear; clc; close all;

%% Question 1
N = 1e6; 

% Gaussian 
data_gauss = randn(N, 1);

% Cauchy with alpha = 1
alpha = 1; 
U = rand(N, 1);
data_cauchy = alpha * tan(pi * U);

% Student's t-distribution with nu = 5
nu5 = 5;
raw_t5 = trnd(nu5, N, 1);
scale_5 = sqrt((nu5 - 2) / nu5); % normalize and rescale
data_t5 = raw_t5 * scale_5;

% Student's t-distribution with nu = 10
nu10 = 10;
raw_t10 = trnd(nu10, N, 1);
scale_10 = sqrt((nu10 - 2) / nu10); % normalize and rescale
data_t10 = raw_t10 * scale_10;

thresh = 4;

frac_gauss = mean(abs(data_gauss) > thresh);
frac_cauchy = mean(abs(data_cauchy) > thresh);
frac_t5 = mean(abs(data_t5) > thresh);
frac_t10 = mean(abs(data_t10) > thresh);

% Display Results
fprintf('Fraction of samples with |x| > 4:\n');
fprintf('Gaussian N(0,1): %.6f\n', frac_gauss);
fprintf('Student''s t (v=10): %.6f\n', frac_t10);
fprintf('Student''s t (v=5): %.6f\n', frac_t5);
fprintf('Cauchy (alpha=1): %.6f\n', frac_cauchy);

figure('Name', 'Heavy Tail Distributions Analysis', 'Position', [100, 100, 1000, 800]);

subplot(4,1,1);
plot(data_gauss);
title('Gaussian N(0,1)');
ylabel('Value');
ylim([-10 10]); % Fixed limit for comparison, though Gaussian stays small

subplot(4,1,2);
plot(data_t10);
title(['Student''s t (\nu = 10, Normalized) - Outliers: ' num2str(frac_t10*100) '%']);
ylabel('Value');
ylim([-15 15]);

subplot(4,1,3);
plot(data_t5);
title(['Student''s t (\nu = 5, Normalized) - Outliers: ' num2str(frac_t5*100) '%']);
ylabel('Value');
ylim([-20 20]);

subplot(4,1,4);
plot(data_cauchy);
title(['Cauchy (\alpha =1 ) - Outliers: ' num2str(frac_cauchy*100) '%']);
ylabel('Value');
ylim([-1000 1000]);
xlabel('Sample Index');

% We see that Cauchy has a MUCH MUCH bigger y-axis scale than the others

%% Question 2

b = conv([1, 0.2], [1, 0.5]); 
a = conv([1, -0.8], [1, -0.7]); 
N = 250;
rng(0); % Set seed for reproducibility
v_t = randn(N, 1); % iid N(0,1)
r_t = filter(b, a, v_t);

analyze_arma_model(r_t, 'Original')

% (e) Pm are the diagonal entries of D, F and L^-1 are identical matrices
% (f) Least Squares lovely! my absolute favorite regression method
% They are almost exactly the same

% (g) The reflection coefficients are significant for the first few orders but
% then quickly decay towards 0, meaning a low order AR model is sufficient

% (h) As order increases, more of the coefficients do fall below the threshold 
% I think by order 5, all are within

%% Question 3

H = [1, -0.99];
a_new = conv(a, H);
r_t_unit = filter(b, a_new, v_t); % Reuse the same parameters etc. from (2)

figure;
plot(r_t_unit);
hold on;
plot(r_t)
title('AR Models Comparison'); 
legend('Unit Root (r_{new})', 'Original (r_t)');
grid on;
xlabel('Time'); 
ylabel('Value');

analyze_arma_model(r_t_unit, 'Unit Root')
% (e) Pm are the diagonal entries of D, F and L^-1 are identical matrices
% (f) They are almost exactly the same

% (g) The reflection coefficients are significant for the first few orders but
% then the absolute value of the coefficients
% quickly decay towards 0, meaning a low order AR model is sufficient

% (h) As order increases, more of the coefficients do fall below the threshold 
% However unlike before, order 10 is not sufficients for all to be below
% the threshold

s_t = diff(r_t_unit);
analyze_arma_model(s_t, 'Difference Series');
% The matrix is still positive definite
% Yes we can see the unit root nonstationarity since in the plot, we see
% the points decay very slowly and in a linear fashion! When we take the
% differenced series, we see the decay to be much faster (similar to the
% original series).

%% Question 4

r_t_student = filter(b, a, data_t5);
analyze_arma_model(r_t_student, 'Original w/ Student T dist.')

r_t_unit_student = filter(b, a_new, data_t5);
analyze_arma_model(r_t_unit_student, 'Unit Root w/ Student T dist.')

s_t_unit_student = diff(r_t_unit_student);
analyze_arma_model(s_t_unit_student, 'Difference w/ Student T dist.')

% All the matrices are still positive definite, which confirms the numerical stability!
% The behavior remains mostly the same (as it should), the only noticeable difference I could
% tell was that the residual autocorrelations decayed much faster and were
% within the thresholds for a much smaller order (order 5) whereas in the
% Unit Root w/ Gaussian run, even order 10 did not get the coefficients within
% the thresholds.


%% Question 5
clear; clc; close all;

% I pulled this data from yfinance using Colab and some processing similar
% to the code in PS2. Then I just imported it from my local directory

financial_data = 'financial_data_2yr_amzn.csv'; % I had to find multiple different datasets so that it would be significant
data_table = readtable(financial_data);

% Extract the prices
prices_sp500 = data_table.x_GSPC;  
prices_nvda = data_table.NVDA;
prices_amzn = data_table.AMZN;

% Compute Log Returns
returns_sp500 = diff(log(prices_sp500));
returns_nvda = diff(log(prices_nvda));
returns_amzn = diff(log(prices_amzn));

% Plotting the returns and square returns for S&P500
figure;
subplot(2,1,1);
plot(returns_sp500);
title('S&P 500 Daily Log Returns');
ylabel('r_t'); 
axis tight;
subplot(2,1,2);
plot(returns_sp500.^2);
title('S&P 500 Squared Returns');
ylabel('(r_t)^2'); 
axis tight;

% Subtract off sample means
returns_sp500 = returns_sp500 - mean(returns_sp500);
returns_nvda = returns_nvda - mean(returns_nvda);
returns_amzn = returns_amzn - mean(returns_amzn);

plot_coeffs(returns_sp500, 'S&P500');
plot_coeffs(returns_nvda, 'NVIDA');
plot_coeffs(returns_amzn, 'AMZN');

% (a)
% Parameters
omega = 0.5;
alpha = 0.6;
beta = 0.4;
T = 1000; % simulation length
rng(0);

r_gauss = zeros(T, 1);
sigma2_gauss = zeros(T, 1);
r_t = zeros(T, 1);
sigma2_t = zeros(T, 1);

% Initializing variance
sigma2_gauss(1) = omega;
sigma2_t(1) = omega;

% Case 1: Gaussian z_t
rng(0);
z_gauss = randn(T, 1);

% GARCH (1,1)
for t = 2:T
    sigma2_gauss(t) = omega + alpha * (r_gauss(t-1)^2) + beta * sigma2_gauss(t-1);
    r_gauss(t) = sqrt(sigma2_gauss(t)) * z_gauss(t); % return
end

sigma_gauss = sqrt(sigma2_gauss);
plot_garch(r_gauss,sigma_gauss, 'Gaussian');

% remade it from before for GARCH
nu = 5;
z_raw = trnd(nu, T, 1);
scale_factor = sqrt((nu-2)/nu); 
z_t = z_raw * scale_factor;

for t = 2:T
    sigma2_t(t) = omega + alpha*(r_t(t-1)^2) + beta*sigma2_t(t-1);
    r_t(t) = sqrt(sigma2_t(t)) * z_t(t);
end

sigma_t = sqrt(sigma2_t);
plot_garch(r_t, sigma_t, 'Student-t');

syn_rt_fit(r_gauss, 'Gaussian', 0.5, 0.6, 0.4);
syn_rt_fit(r_t, 'Student-t', 0.5, 0.6, 0.4);

% As expected in the Gaussian case, the volatility creates a tight envelope
% and tracks the variance effectively. However, the Student-t introduces heavy tails (outliers). 
% It is still decent but since the model assumes Gaussian errors, it interprets these outliers as high-variance events, 
% causing the estimated volatility to overreact and spike aggressively compared to the smoother Gaussian case.

% (b)
real_rt_fit(returns_sp500, 'S&P 500');
real_rt_fit(returns_nvda,  'NVIDIA');
real_rt_fit(returns_amzn,  'Amazon');

% GARCH(1,1) does not envelope the signal completely. However it
% follows the general trend and contains most of the underlying signal.
% However ARCH(2) follows the signal much better, spiking more rapidly
% especially in the SP500 and AMZN cases.

%% Functions (there was so much repetition it had to have been done)

function real_rt_fit(returns_data, label)
    fprintf("%s r_t Fitting ", label);
    % Fit GARCH(1,1)
    Mdl_G11 = garch(1,1);
    est_G11 = estimate(Mdl_G11, returns_data, 'Display', 'params');
    v_G11 = infer(est_G11, returns_data);
    sigma_G11 = sqrt(v_G11);

    % Fit ARCH(2)
    Mdl_ARCH2 = garch(0,2);
    est_ARCH2 = estimate(Mdl_ARCH2, returns_data, 'Display', 'params');
    v_ARCH2 = infer(est_ARCH2, returns_data);
    sigma_ARCH2 = sqrt(v_ARCH2);

    % Plotting: GARCH(1,1) Envelope
    figure;
    plot(returns_data, 'Color', [0.7 0.7 0.7]); 
    hold on;
    plot(sigma_G11, 'b', 'LineWidth', 1.5);
    plot(-sigma_G11, 'b', 'LineWidth', 1.5);
    hold off;
    title(sprintf('GARCH(1,1) Returns vs Estimated Volatility (%s)', label));
    legend('Returns', 'Est. \sigma_t (GARCH)', 'Location', 'Best');
    ylabel('Time'); 
    ylabel('Magnitude'); 
    axis tight; 
    grid on;

    % Plotting: ARCH(2) Envelope
    figure;
    plot(returns_data, 'Color', [0.7 0.7 0.7]); 
    hold on;
    plot(sigma_ARCH2, 'r', 'LineWidth', 1.5);
    plot(-sigma_ARCH2, 'r', 'LineWidth', 1.5);
    hold off;
     title(sprintf('ARCH(2) Returns vs Estimated Volatility (%s)', label));
    legend('Returns', 'Est. \sigma_t (ARCH)', 'Location', 'Best');
    ylabel('Magnitude'); 
    axis tight; 
    grid on;
end

function syn_rt_fit(returns_data, label, omega, alpha, beta)
    % Fit to GARCH(1,1)
    Mdl_G11 = garch(1,1);
    est_G11 = estimate(Mdl_G11, returns_data, 'Display', 'off');
   
    Parameters = {'Omega'; 'Alpha'; 'Beta'};
    True_Values = [omega; alpha; beta];
    Estimated   = [est_G11.Constant; est_G11.ARCH{1}; est_G11.GARCH{1}];
    Diff        = abs(True_Values - Estimated);
    
    resultsTable = table(Parameters, True_Values, Estimated, Diff);
    disp(resultsTable);

    Mdl_ARCH2 = garch(0,2); % ARCH(2) is equivalent to GARCH(0,2)
    est_ARCH2 = estimate(Mdl_ARCH2, returns_data, 'Display', 'off');
    
    % Infer volatility
    v_inferred = infer(est_ARCH2, returns_data);
    sigma_inferred = sqrt(v_inferred);
    
    figure;
    plot(returns_data, 'Color', [0.6 0.6 0.6]); 
    hold on; 
    plot(sigma_inferred, 'b', 'LineWidth', 1.5);         
    plot(-sigma_inferred, 'b', 'LineWidth', 1.5);       
    hold off;
    title(sprintf('ARCH(2) Returns vs Estimated Volatility (%s)', label));
    legend('Returns (r_t)', 'Estimated \sigma_t', 'Location', 'Best');
    ylabel('Magnitude');
    xlabel('Time');
    axis tight; 
    grid on;
end

function plot_garch(r_t, sigma_t, label)
    figure;
    plot(r_t, 'Color', [0.6 0.6 0.6]); 
    hold on;
    plot(sigma_t, 'r', 'LineWidth', 1.5);
    plot(-sigma_t, 'r', 'LineWidth', 1.5);
    title(sprintf('r_t fit to GARCH(1,1) (%s)', label));
    legend('Returns (r_t)', 'Volatility Envelope (\sigma_t)');
    hold off;
end

function plot_coeffs(returns, label)
    num_lags = 20;
    figure;
    subplot(2,1,1);
    autocorr(returns, 'NumLags', num_lags);
    title(sprintf('Sample Autocorrelation: Log Returns (%s)', label));
    xlabel('Lag'); 
    ylabel('Autocorrelation');
    
    subplot(2,1,2);
    autocorr(returns.^2, 'NumLags', num_lags);
    title(sprintf('Sample Autocorrelation: Squared Returns (%s)', label));
    xlabel('Lag'); 
    ylabel('Autocorrelation');
end

function analyze_arma_model(r_t, label)
    fprintf('\n \nAnalysis: %s \n', label)
    N = length(r_t); 

    % (a)
    M = 10;
    [rho, lags] = autocorr(r_t, 'NumLags', M);
    figure;
    stem(lags, rho, 'filled'); 
    title(sprintf('Sample Autocorrelation (%s)', label));
    xlabel('Lag (m)'); 
    ylabel('\rho(m)');
    hold on;
    ylim([-1.2, 1.2]);
    yline(0.2, '--r'); 
    yline(-0.2, '--r'); 
    hold off;

    % (b)
    C = toeplitz(rho); % (M+1)x(M+1) matrix 
    eigenvalues = eig(C);
    fprintf('Smallest Eigenvalue of C: %e\n', min(eigenvalues));
    if all(eigenvalues > 0)
        disp('Result: Positive Definite');
    else
        disp('Result: Not Positive Definite');
    end

    % (c) 
    % chol returns C = R' * R where R is lower triangular with 'lower'
    [L_chol, ~] = chol(C, 'lower'); 
    D_diag = diag(L_chol).^2; % Extract variances (diagonal squared)
    L = L_chol * diag(1 ./ diag(L_chol)); % Normalize columns to get 1 on diagonal

    % (d)
    gamma_0 = var(r_t, 1); 
    gamma_vals = rho * gamma_0;
    F = eye(M+1); 
    P = zeros(M+1, 1); % Store prediction error powers
    reflect = zeros(M, 1); 
    P(1) = gamma_vals(1); % Order 0
    
    for p = 1:M
        [A_p, E_p, K_p] = levinson(gamma_vals, p);
        P(p+1) = E_p;
        reflect(p) = K_p(end); 
        F(p+1, 1:p+1) = fliplr(A_p);
    end

    % (e)
    FCF = F * C * F'; % Compute FCF'
    L_inv = inv(L);
    
    fprintf('\n(e) Decomposition Verification (First 5):\n');
    fprintf('Prediction Powers (P_m): %f %f %f %f %f\n', P(1:5));
    fprintf('Diagonals of D (LDL''): %f %f %f %f %f\n', D_diag(1:5));
    fprintf('First col of L^-1 (should = F): %f %f %f %f %f\n', L_inv(1:5));

    % (f)
    X_ls = zeros(N-M, M);
    y_ls = r_t(M+1:N);
    for k = 1:M
        X_ls(:, k) = r_t(M+1-k:N-k);
    end
    w_ls = (X_ls' * X_ls) \ (X_ls' * y_ls);
    ar_ls = [1; -w_ls];
    ar_lev = fliplr(F(M+1, :))'; 
    
    fprintf('\n Part (f) \n');
    fprintf(' Lag |   Levinson  | Least Squares\n');
    for k = 1:5
        fprintf(' %2d  |  %10.6f |  %10.6f\n', k-1, ar_lev(k), ar_ls(k));
    end

    % (g)
    fprintf('\n Part (g) \n');
    fprintf(' Order | Reflect Coeff | Pred Power\n');
    for k = 1:length(reflect)
        fprintf('  %2d   |   %8.5f    |  %8.5f\n', k, reflect(k), P(k+1));
    end

    % (h) 
    orders_to_test = [2, 5, 10]; 
    figure;
    for i = 1:length(orders_to_test)
        k_ord = orders_to_test(i);
        
        % Use Levinson for consistency; row k_ord+1, flip back to get [1, a1...]
        coeffs_k = fliplr(F(k_ord+1, 1:k_ord+1));
        
        residuals = filter(coeffs_k, 1, r_t);
        
        [rho_res, lags_res] = xcorr(residuals, 20, 'biased');
        mid = find(lags_res == 0);
        rho_res = rho_res(mid:end) / rho_res(mid); % Normalize
        
        subplot(3,1,i);
        stem(0:20, rho_res, 'filled');
        title(sprintf('Residual Autocorrelation (AR Order %d) - %s', k_ord, label));
        ylim([-1.2, 1.2]);
        yline(0.2, '--r'); 
        yline(-0.2, '--r');
        ylabel('\rho');
    end
    xlabel('Lag');
end