function [best_cost,best_V,Convergence_curve]=VDO(SearchAgents_no,MaxFEs,lb,ub,dim,fobj)
% VDO: Virus Diffusion Optimizer
% 输入:
%   SearchAgents_no - 种群大小
%   MaxFEs - 最大函数评估次数
%   lb, ub - 下/上界（向量或标量）
%   dim - 维度
%   fobj - 目标函数
% 输出:
%   best_cost, best_V, Convergence_curve

%% ---------------- Initialization ----------------
FEs = 0;
best_cost = inf;
best_V = zeros(1,dim);

if isscalar(lb), lb = lb*ones(1,dim); end
if isscalar(ub), ub = ub*ones(1,dim); end

% Initialize population
V = initialization(SearchAgents_no, dim, ub, lb); 
costs = inf(1,SearchAgents_no);

for i=1:SearchAgents_no
    costs(i) = fobj(V(i,:));
    FEs = FEs + 1;
    if costs(i) < best_cost
        best_cost = costs(i);
        best_V = V(i,:);
    end
end

Convergence_curve = [];
it = 1;
rec = 1;

%% -------- HSV-inspired Parameters --------
burst_rate = 2;
decay_exp = 1.5; 

tropism_min = 0.05;
tropism_max = 0.3;
divide_num = max(1, floor(dim/4)); % 分割维度

latency_depth = 10;
latent_jump_base = 0.02;
latent_jump_extra = 0.08;

envelope_flip_flags = [1, -1]; % 方向翻转，模拟病毒出芽方向

%% History buffers for latency
newV = zeros(SearchAgents_no, dim);
newV_cost = inf(1, SearchAgents_no);
rV = zeros(SearchAgents_no, dim, latency_depth);
rV_cos = inf(SearchAgents_no, latency_depth);

%% record first generation
rV(:,:,rec) = V;
rV_cos(:,rec) = costs';
rec = rec + 1;

%% ---------------- Main Loop ----------------
while FEs < MaxFEs
    % Adaptive replication step
    burst_factor = burst_rate * (1 - FEs/MaxFEs)^decay_exp;
    tropism_prob = tropism_max - (tropism_max - tropism_min) * (FEs/MaxFEs);
    
    infection_angle = 2 * atan(1 - FEs/MaxFEs);
    explore_intensity = 2 * log(1/rand) * infection_angle;
    
    envelope_flip = envelope_flip_flags(randi(2));
    
    calPositions = V;
    div_dims = randperm(dim);
    for j = 1:max(divide_num,1)
        th = best_V(div_dims(j));
        index = calPositions(:,div_dims(j)) > th;
        if sum(index) < size(calPositions,1)/2
            index = ~index;
        end
        calPositions = calPositions(index, :);
        if isempty(calPositions), calPositions = V; break; end
    end
    
    D = best_V - calPositions;
    D_mean = mean(D,1);
    beta_t = (1 - FEs/MaxFEs) ^ (2 * FEs/MaxFEs);
    
    beta = size(calPositions, 1) / SearchAgents_no;
    gama = 1 / sqrt(1 - beta^2 + eps);
    if rand() > 0.5
        step = burst_factor * (rand(size(D_mean)) - 0.5) * beta_t * sin(2*pi*rand());
    else
        step = burst_factor * (rand(size(D_mean)) - 0.5) * beta_t * cos(2*pi*rand());
    end
    step2 = 0.1 * burst_factor * (rand(size(D_mean)) - 0.5) * beta_t * (1 + 1/2*(1 + tanh(beta/gama))*beta_t);
    step3 = 0.1 * (rand() - 0.5) * beta_t;
    
    act_prob = 1./(1 + exp(- (0.5 - 10*rand(size(D_mean)))));
    act = actCal(act_prob);
    
    %% Update each virion
    for i = 1:SearchAgents_no
        newV(i,:) = V(i,:);
        
        if rand() > tropism_prob
            newV(i,:) = newV(i,:) + envelope_flip .* step .* D_mean;
        else
            newV(i,:) = newV(i,:) + envelope_flip .* step2 .* D_mean;
        end
        
        if rand() < 0.8
            if rand() > 0.5
                newV(i, div_dims(1)) = best_V(div_dims(1)) + step3 * D_mean(div_dims(1));
            else
                newV(i,:) = (1 - act) .* newV(i,:) + act .* best_V;
            end
        else
            newV(i,:) = ub + lb - newV(i,:);
        end
        
        if rand() < 0.08
            radius = sqrt(sum((best_V - newV(i,:)).^2));
            r3 = rand();
            spiral = radius * (sin(2*pi*r3) + cos(2*pi*r3));
            newV(i,:) = best_V + envelope_flip .* randn(1,dim) .* spiral * rand();
        end
        
        levy_prob = latent_jump_base + latent_jump_extra * (FEs/MaxFEs);
        if explore_intensity <= 1 && rand() < levy_prob
            newV(i,:) = newV(i,:) + levy(dim) .* (best_V - newV(i,:));
        elseif rand() < 0.05
            newV(i,:) = newV(i,:) + 0.01*levy(dim) .* D_mean;
        end
        
        if rand() < 0.12
            ids = randperm(SearchAgents_no,3);
            r1 = V(ids(1),:); r2 = V(ids(2),:); r3 = V(ids(3),:);
            F_de = 0.6; CR = 0.9;
            trial = r1 + F_de * (r2 - r3);
            jrand = randi(dim);
            for jj = 1:dim
                if rand() <= CR || jj == jrand
                    newV(i,jj) = trial(jj);
                end
            end
        end
        
        Flag4ub = newV(i,:) > ub;
        Flag4lb = newV(i,:) < lb;
        newV(i,:) = (newV(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
        
        newV_cost(i) = fobj(newV(i,:));
        FEs = FEs + 1;
        
        rV(i,:,rec) = newV(i,:);
        rV_cos(i, rec) = newV_cost(i);
        
        if newV_cost(i) < costs(i)
            V(i,:) = newV(i,:);
            costs(i) = newV_cost(i);
        end
        if newV_cost(i) < best_cost
            best_cost = newV_cost(i);
            best_V = newV(i,:);
        end
        
        if FEs >= MaxFEs
            break;
        end
    end
    
    rec = rec + 1;
    
    if rec > latency_depth || FEs >= MaxFEs
        [lcost, Iindex] = min(rV_cos, [], 2);
        for i = 1:SearchAgents_no
            V(i,:) = rV(i, :, Iindex(i));
            costs(i) = lcost(i);
        end
        rec = 1;
    end
    
    Convergence_curve(it) = best_cost;
    it = it + 1;
    
end % while

end % function

%% ---------------- Helper functions ----------------

function [act] = actCal(X)
    % 将概率向量 X 二值化为 0/1
    act = X;
    act(act >= 0.5) = 1;
    act(act < 0.5) = 0;
end

function L = levy(dim)
    % Lévy flight generator (返回 1 x dim)
    beta = 1.5;
    sigma_u = ( gamma(1+beta) * sin(pi*beta/2) / ...
               ( gamma((1+beta)/2) * beta * 2^((beta-1)/2) ) )^(1/beta);
    u = randn(1,dim) * sigma_u;
    v = randn(1,dim);
    step = u ./ (abs(v).^(1/beta) + eps);
    L = step;
end
