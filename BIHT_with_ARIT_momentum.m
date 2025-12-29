function [x,BIHT_nbiter,BIHT_l2_err,BIHT_Hamming_err] = BIHT_with_ARIT_momentum(x0,M,N,K,maxiter)
    %% 初始化
    htol = 0.01 * M;
    Phi = randn(M,N)/sqrt(M);
    A = @(in) sign(Phi*in);
    
    y = A(x0);
    
    % 更好的初始化：使用观测一致投影
    x_init = Phi' * y;
    [~, init_idx] = sort(abs(x_init), 'descend');
    x = zeros(N,1);
    x(init_idx(1:min(2*K, N))) = x_init(init_idx(1:min(2*K, N)));
    x = x / norm(x);
    
    eta = norm(Phi'*Phi, 2);
    mu = 0.8;  % 初始步长
    
    % 动量参数
    beta = 0.9;  % 动量系数
    velocity = zeros(N,1);  % 动量速度项
    
    ii = 0;
    hd = Inf;
    best_x = x;
    best_hd = Inf;
    stagnation_count = 0;
    
    % ARIT参数 - 经过调优的值
    c = 0.8;
    k_values = [2.0, 1.5, 1.0, 0.8, 0.5];  % 多组k值尝试
    
    while ii < maxiter
        % 自适应响应阈值参数
        if ii < length(k_values)
            k = k_values(ii+1);
        else
            k = 0.5;  % 后期使用较小的k
        end
        
        % 计算梯度
        sign_diff = y - A(x);
        g = Phi' * sign_diff;
        
        % 带动量项的梯度更新
        velocity = beta * velocity + (1 - beta) * g;
        g_momentum = velocity;  % 使用动量调整后的梯度
        
        % 应用Nesterov动量（更先进的动量技术）
        if ii > 0
            % Nesterov: 先看一步再计算梯度
            x_lookahead = x + beta * velocity;
            sign_diff_lookahead = y - A(x_lookahead);
            g_nesterov = Phi' * sign_diff_lookahead;
            
            % 结合两种动量
            a = x + mu * (0.7 * g_momentum + 0.3 * g_nesterov) / eta;
        else
            a = x + mu * g_momentum / eta;
        end
        
        % ARIT阈值操作
        lambda = k * eta / c^2;
        x_new = zeros(size(a));
        
        non_zero_count = 0;
        for idx = 1:length(a)
            u_val = a(idx);
            abs_u = abs(u_val);
            threshold = lambda * c / (2 * eta);
            
            if abs_u > threshold
                % 数值稳定的ARIT计算
                p_val = 1/(3*c^2) - (abs_u^2)/9;
                q_val = lambda/(4*eta*c) - abs_u/(3*c^2) - (abs_u^3)/27;
                
                % 确保数值稳定性
                if abs(p_val) < 1e-12
                    p_val = 1e-12 * sign(p_val);
                end
                
                r = sign(q_val) * sqrt(abs(p_val));
                denom = r^3;
                if abs(denom) < 1e-12
                    denom = 1e-12 * sign(denom);
                end
                
                arg_val = q_val / denom;
                
                if p_val < -1e-8
                    arg_val = max(min(arg_val, 1e6), 1+1e-8);  % 限制范围
                    theta = acosh(arg_val);
                    h_bar = -2 * r * cosh(theta/3) + abs_u/3;
                elseif p_val > 1e-8
                    theta = asinh(arg_val);
                    h_bar = -2 * r * sinh(theta/3) + abs_u/3;
                else
                    % p接近0的情况
                    h_bar = sign(q_val) * (abs(2*q_val))^(1/3) + abs_u/3;
                end
                
                x_new(idx) = sign(u_val) * max(0, real(h_bar));
                non_zero_count = non_zero_count + 1;
            else
                x_new(idx) = 0;
            end
        end
        
        % 强制稀疏性：如果非零元素太多，进行硬阈值
        if non_zero_count > 3*K
            [~, sort_idx] = sort(abs(x_new), 'descend');
            temp = x_new;
            temp(sort_idx(2*K+1:end)) = 0;
            x_new = temp;
        end
        
        % 自适应步长调整
        new_hd = nnz(y - A(x_new));
        if new_hd >= hd
            mu = mu * 0.8;  % 减少步长
            stagnation_count = stagnation_count + 1;
            
            % 当停滞时，减小动量系数
            if stagnation_count > 5
                beta = max(0.5, beta * 0.95);
            end
        else
            mu = min(mu * 1.05, 1.5);  % 适度增加步长
            stagnation_count = 0;
            beta = min(0.99, beta * 1.01);  % 恢复动量系数
        end
        
        % 如果停滞多次，尝试随机扰动
        if stagnation_count > 20
            perturbation = 0.05 * randn(N,1) / (ii+1);
            x_new = x_new + perturbation;
            x_new = x_new / norm(x_new);
            stagnation_count = 0;
            mu = 1.0;  % 重置步长
            beta = 0.9;  % 重置动量系数
        end
        
        x = x_new;
        hd = new_hd;
        
        % 保存最佳结果
        if hd < best_hd
            best_hd = hd;
            best_x = x;
        end
        
        ii = ii + 1;
        
        % 提前终止检查
        if hd <= htol
            fprintf('提前收敛于迭代 %d，汉明误差: %d/%d\n', ii, hd, M);
            break;
        end
        
        % 显示进度
        if mod(ii, 100) == 0
            fprintf('迭代 %d: 汉明误差=%d/%d, 非零元素=%d, 步长=%.3f, 动量系数=%.3f\n', ...
                    ii, hd, M, non_zero_count, mu, beta);
        end
    end
    
    % 使用最佳结果并投影到球面
    x = best_x / norm(best_x);
    
    BIHT_nbiter = ii;
    BIHT_l2_err = norm(x0 - x)/norm(x0);
    BIHT_Hamming_err = (nnz(y - A(x)))/M;
    
    fprintf('最终结果: 迭代次数=%d, 相对误差=%.4f, 汉明误差=%.4f\n', ...
            BIHT_nbiter, BIHT_l2_err, BIHT_Hamming_err);
end