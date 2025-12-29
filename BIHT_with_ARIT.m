function [x,BIHT_nbiter,BIHT_l2_err,BIHT_Hamming_err] = BIHT_with_ARIT(x0,M,N,K,maxiter)
    %% 初始�?
    htol = 0.01 * M;
    Phi = randn(M,N)/sqrt(M);
    A = @(in) sign(Phi*in);
    
    y = A(x0);
    
    % 更好的初始化：使用观测一致�?投影
    x_init = Phi' * y;
    [~, init_idx] = sort(abs(x_init), 'descend');
    x = zeros(N,1);
    x(init_idx(1:min(2*K, N))) = x_init(init_idx(1:min(2*K, N)));
    x = x / norm(x);
    
    eta = norm(Phi'*Phi, 2);
    mu = 0.8;  % 初始步长
    
    ii = 0;
    hd = Inf;
    best_x = x;
    best_hd = Inf;
    stagnation_count = 0;
    
    % ARIT参数 - 经过调优的�?
    c = 0.8;
    k_values = [2.0, 1.5, 1.0, 0.8, 0.5];  % 多组k值尝�?
    
    while ii < maxiter
        % 自�?应�?择k�?
        if ii < length(k_values)
            k = k_values(ii+1);
        else
            k = 0.5;  % 后期使用较小的k
        end
        
        % 计算梯度
        sign_diff = y - A(x);
        g = Phi' * sign_diff;
        
        % 带动量项的梯度步
        if ii > 0
            momentum = 0.3;
            a = x + mu * g / eta + momentum * (x - prev_x);
        else
            a = x + mu * g / eta;
        end
        prev_x = x;
        
        % ARIT阈�?操作
        lambda = k * eta / c^2;
        x_new = zeros(size(a));
        
        non_zero_count = 0;
        for idx = 1:length(a)
            u_val = a(idx);
            abs_u = abs(u_val);
            threshold = lambda * c / (2 * eta);
            
            if abs_u > threshold
                % 数�?稳定的ARIT计算
                p_val = 1/(3*c^2) - (abs_u^2)/9;
                q_val = lambda/(4*eta*c) - abs_u/(3*c^2) - (abs_u^3)/27;
                
                % 确保数�?稳定�?
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
                    % p接近0的情�?
                    h_bar = sign(q_val) * (abs(2*q_val))^(1/3) + abs_u/3;
                end
                
                x_new(idx) = sign(u_val) * max(0, real(h_bar));
                non_zero_count = non_zero_count + 1;
            else
                x_new(idx) = 0;
            end
        end
        
        % 强制�?��性：如果非零元素太多，进行硬阈�?
        if non_zero_count > 3*K
            [~, sort_idx] = sort(abs(x_new), 'descend');
            temp = x_new;
            temp(sort_idx(2*K+1:end)) = 0;
            x_new = temp;
        end
        
        % 更新步长（自适应调整�?
        new_hd = nnz(y - A(x_new));
        if new_hd >= hd
            mu = mu * 0.8;  % 减少步长
            stagnation_count = stagnation_count + 1;
        else
            mu = min(mu * 1.05, 1.5);  % 适度增加步长
            stagnation_count = 0;
        end
        
        % 如果停滞多次，尝试随机扰�?
        if stagnation_count > 20
            perturbation = 0.1 * randn(N,1) / (ii+1);
            x_new = x_new + perturbation;
            x_new = x_new / norm(x_new);
            stagnation_count = 0;
            mu = 1.0;  % 重置步长
        end
        
        x = x_new;
        hd = new_hd;
        
        % 保存�?��结果
        if hd < best_hd
            best_hd = hd;
            best_x = x;
        end
        
        ii = ii + 1;
        
        % 提前终止�?��
        if hd <= htol
            break;
        end
        
        % 显示进度
        if mod(ii, 200) == 0
            fprintf('Iter %d: Hamming=%d/%d, non-zero=%d\n', ii, hd, M, non_zero_count);
        end
    end
    
    % 使用�?��结果并投影到球面
    x = best_x / norm(best_x);
    
    BIHT_nbiter = ii;
    BIHT_l2_err = norm(x0 - x)/norm(x0);
    BIHT_Hamming_err = (nnz(y - A(x)))/M;
end