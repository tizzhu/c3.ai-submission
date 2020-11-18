clc
clear

% Load initial data from MAT files
load factor231.mat

data_in_1 = factor231;
data_in_dep_1 = double(benchmark);

% size of selected factors
sg_size = 2;
% Cross validation threshold
threshold = sum(data_in_dep_1==0)/length(data_in_dep_1);

% Set keys for working with different steps of algorithm
key_step_1 = 0; % Set key_step_1 = 1 to work with the first step of the problem
key_step_2 = 1; % Set key_step_2 = 1 to work with the second step of the problem
key_step_3 = 0; % Set key_step_3 = 1 to work with the third step of the problem
key_step_4 = 0; % Set key_step_4 = 1 to work with the fourth step of the problem

matrix_splined_factors = data_in_1;

if key_step_2 == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STEP 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Calculation of the importance of splined factors
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vars_splined_factors=cell(size(matrix_splined_factors,2),1);
    for i=1:size(matrix_splined_factors,2)
        vars_splined_factors{i} = sprintf('x%g',i);
    end
    
    
    % Create problem statement
    clear  toolboxstruc_arr  problem_statement
    problem_statement_1=sprintf('%s\n',...
        'maximize',...
        '  logexp_sum(matrix_splined_factors)',...
        'Solver: VAN, precision =9',...
        '',...
        'Calculate',...
        'Point: point_problem_1',...
        '  logistic(matrix_splined_factors)',...
        '');
    
    CVconst = 10;
    problem_statement_2=sprintf('%s\n',...
        ['for {matrix_fact_in; matrix_fact_out; #n}=CrossValidation(',mat2str(CVconst),', matrix_splined_factors) '],...
        'Problem: problem_Logistic_Regression_CV_#n, type = maximize',...
        'Objective: objective_max_likelihood',...
        '  logexp_sum(matrix_fact_in)',...
        'Solver: VAN, precision=9',...
        '',...
        'Calculate',...
        'point: point_problem_Logistic_Regression_CV_#n',...
        '  logexp_sum_1_#n(matrix_fact_out)',...
        '  logistic_in_#n(matrix_fact_in)',...
        '  logistic_out_#n(matrix_fact_out)',...
        'end for',...
        '');
    
    
    %     all possible subgroups combination
    sg_comb = nchoosek(1:size(matrix_splined_factors,2), sg_size);
    comb_num = size(sg_comb,1);
    factors_bruteforce_result = struct('combination',cell(1,comb_num),'data',cell(1,comb_num),...
        'objective',cell(1,comb_num),'coeffs',cell(1,comb_num),'p_value',cell(1,comb_num),'logistic',cell(1,comb_num));

    
    
    tic
    parfor k = 1:comb_num
        %Pack matrix with splined factors left from the previous step of cycle to
        %PSG Toolbox structure

        toolboxstruc_arr_1=tbpsg_matrix_pack('matrix_splined_factors',...
            [ones(size(matrix_splined_factors,1),1),matrix_splined_factors(:,sg_comb(k,:))],...
            ['alpha';vars_splined_factors(sg_comb(k,:))],data_in_dep_1(1:end));
        
        %         toolboxstruc_arr_1=tbpsg_matrix_pack('matrix_splined_factors',...
        %             [ones(size(matrix_splined_factors,1),1),matrix_splined_factors(:,sg_comb(k,:))],...
        %             vars_splined_factors,[zeros(18,1);ones(13,1)]);
        
        
        % Uncomment the following line to open the problem in Toolbox Window
        %tbpsg_toolbox(problem_statement,toolboxstruc_arr);
        
        % Optimize problem
        [solution_str_1,outargstruc_arr_1] = tbpsg_run(problem_statement_1, toolboxstruc_arr_1);
        
        % Save solution of the optimization problem to output structure
        [output_structure_1] = tbpsg_solution_struct(solution_str_1, outargstruc_arr_1);
        
        point_data_loc = tbpsg_optimal_point_data(solution_str_1, outargstruc_arr_1);
        point_data = point_data_loc.problem_1;
        point_var_loc = tbpsg_optimal_point_vars(solution_str_1, outargstruc_arr_1);
        point_var = point_var_loc.problem_1;
        
        mdl = fitglm(matrix_splined_factors(:,sg_comb(k,:)),data_in_dep_1(1:end),'Distribution','binomial');
        
        
        factors_bruteforce_result(k).combination = sg_comb(k,:);
        factors_bruteforce_result(k).data = matrix_splined_factors(:,sg_comb(k,:));
        %         factors_bruteforce_result(k).vars = ['alpha';vars_splined_factors(sg_comb(k,:))];
        factors_bruteforce_result(k).objective = output_structure_1.objective(1);
        factors_bruteforce_result(k).coeffs = point_data;
        factors_bruteforce_result(k).logistic = output_structure_1.vector_data;
        factors_bruteforce_result(k).p_value = mdl.Coefficients.pValue;

        % Optimize problem
        [solution_str_2,outargstruc_arr_2] = tbpsg_run(problem_statement_2, toolboxstruc_arr_1);
        % Save solution of the optimization problem to output structure
        [output_structure_2] = tbpsg_solution_struct(solution_str_2, outargstruc_arr_2);
        
        benchmark_loc = data_in_dep_1;
        TP=0;FP=0;TN=0;FN=0;ce_in=[];tp_in=[];tn_in=[];count=0;
        
        for jk=1:CVconst
            field_in = sprintf('logistic_in_%g',jk);
            logistic_in = output_structure_2.vector_data.(field_in);
            %output_structure.vector_data{2*jk};
            field_out = sprintf('logistic_out_%g',jk);
            logistic_oos = output_structure_2.vector_data.(field_out);
            loc = size(logistic_oos,1);
            benchmark_oos = benchmark_loc(1:loc);
            benchmark_loc(1:loc) = [];
            
            benchmark_in = data_in_dep_1;
            benchmark_in(count+1:count+loc) = [];
            count= count+loc;
            
            logistic_in_threshold = (logistic_in > threshold);
            ce_in(jk) = sum(logistic_in_threshold ~= benchmark_in)/length(benchmark_in);
            tp_in(jk) = sum(benchmark_in(logistic_in_threshold == 1) == 1)/ sum(benchmark_in == 1);
            tn_in(jk) = sum(benchmark_in(logistic_in_threshold == 0) == 0)/ sum(benchmark_in == 0);
            
            for mk = 1:length(benchmark_oos)
                if logistic_oos(mk) <= threshold
                    if benchmark_oos(mk) ==0
                        TN=TN+1;
                    else
                        FN=FN+1;
                    end
                end
                if logistic_oos(mk) > threshold
                    if benchmark_oos(mk) ==1
                        TP=TP+1;
                    else
                        FP=FP+1;
                    end
                end
            end
        end

        factors_bruteforce_result(k).ce_in = mean(ce_in);
        factors_bruteforce_result(k).tp_in = mean(tp_in);
        factors_bruteforce_result(k).tn_in = mean(tn_in);
        factors_bruteforce_result(k).ce_cv = (FN+FP)/length(data_in_dep_1);
        factors_bruteforce_result(k).tp_cv = TP/(TP+FN);
        factors_bruteforce_result(k).tn_cv = TN/(TN+FP);
    end
    
    
    toc
    
    T = struct2table(factors_bruteforce_result);
    sortedT = sortrows(T, 'objective','descend');
    factors_sorted_result = table2struct(sortedT);
    
    save factors_bruteforce_result.mat factors_bruteforce_result
    save factors_sorted_result.mat factors_sorted_result
end