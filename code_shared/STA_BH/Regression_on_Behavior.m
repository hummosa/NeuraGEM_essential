%% 1. settings

clear
path('helper/',path)

loadpath        = 'data/';

% read data names
files = dir ([loadpath '*.mat']);

% Build Array (VP) with Names and Filenames
for i = 1:length(files)
    VP(1,i) = {strtok(files(i).name,'.')};
    VP(2,i) = {strtok(files(i).name)};
end


%% 2. set up data for modeling and run GLMs
for m2r = 1:2 %models to run (m2r)
    ai=0;
    for a = 1:size(VP,2) %
        ai = ai + 1;
        
        
        %load data
        load([loadpath,VP{2,a}]);

        %extract factors 
        FN = fieldnames(all);
        for c = 1 : length(FN)
            eval([FN{c} '= [all.(FN{c})];']);
        end;

        %some recoding (see description of factors below)
        
        ACC = Error*-1+1; % 0 = error; 1 = correct
        Error(Error==0)=-1;
        PrevError = Error_Tm1;
        PrevError(PrevError==0)=-1;
        Distance(Distance==0)=-1;
        

        % The individual factors are: Congruence = incongruence between flanker and
        % target (−1 = congruent, 1 = incongruent), Error = accuracy (−1 = correct, 1 =
        % error), Distance = distance between flanker and target (−1 = close, 1 = far), RSI =
        % response-stimulus interval from the previous response (−1 = short = 250 ms, 1 =
        % long = 700 ms), PrevError = accuracy of immediately preceding trial
        % (−1 = correct, 1 = error), Trnr = logscaled
        % trial number (reflecting the time on the task)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% Accuracy models %%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        switch m2r

            case 1
               ModN = 'Accuracy_Model';%name of the model
               M.DesignM = nanrem(table(Congruence, Distance, RSI ... 
                    , PrevError, Trnr,ACC));%define design matrix
                M.labels = { {'cong' 'incong'} {'close' 'far'} {'short' 'long'} {'prevCcorrect' 'prevError'} {}};
                M.CatVars =  1 : 4; %what are the categorical variabels
                M.modelspec = ['ACC ~ Congruence + Distance + RSI + PrevError + Trnr'];%specification of the model
                M.modelspec = [M.modelspec '+ PrevError:Congruence'];%add interaction
                SP = 'results/BehavioralRegression/';%save path
                %LF = 'normal';
                LF = 'binomial'; %distibution of the DV
                path_reg_m = [SP ModN '/']; %combine path for output and the name of this model
                if ~exist(path_reg_m) %if path is nonexist, we create it
                    mkdir(path_reg_m);
                end

                if a == 1
                    AllM.(ModN).CatVars = M.CatVars; AllM.(ModN).modelspec = M.modelspec;
                    path_reg_m = [SP ModN '/']; %combine path for output and the name of this model
                    if ~exist(path_reg_m) %if path is nonexist, we create it
                        mkdir(path_reg_m);
                    end
                end

                SavePaths{m2r} = path_reg_m;


           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%%%%%%%%%%%%%%% RT models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            case 2
                ModN = 'RT_Model';
                M.DesignM = nanrem(table(Congruence, Error, Distance,RSI ... 
                    , PrevError, Trnr,LogRT));
                M.labels = { {'cong' 'incong'} {'correct' 'error'} {'close' 'far'} {'short' 'long'} {'prevCorrect' 'prevError'} {}};%{'short' 'long'} {'close' 'far'}
                M.CatVars =  1 : 5;
                M.modelspec = ['LogRT ~ Congruence + Error + Distance + RSI +'  ...
                    ' PrevError  +  Trnr'];
                M.modelspec = [M.modelspec '+ Congruence:PrevError'];%add interaction
                SP = 'results/BehavioralRegression/';
                LF = 'normal';
                %LF = 'binomial';
                path_reg_m = [SP ModN '/']; %combine path for output and the name of this model
                if ~exist(path_reg_m) %if path is nonexist, we create it
                    mkdir(path_reg_m);
                end

                if a == 1
                    AllM.(ModN).CatVars = M.CatVars; AllM.(ModN).modelspec = M.modelspec;
                    path_reg_m = [SP ModN '/']; %combine path for output and the name of this model
                    if ~exist(path_reg_m) %if path is nonexist, we create it
                        mkdir(path_reg_m);
                    end
                end

                SavePaths{m2r} = path_reg_m;
        end

        a

        % calculations

        RegMod1     = fitglm(M.DesignM, M.modelspec, 'CategoricalVars', M.CatVars,'Distribution', LF);

        DataByGLM   = AGFHK_Eval_Model(M.DesignM, M.modelspec, M.labels);
        [AllM]      = AGFHK_Model_Details( AllM, RegMod1, M, DataByGLM, ai, ModN );
    end
end

%% 3. plot the resutls
N = fieldnames(AllM);
for mc = 1 : length(N)

    AGFHK_summary(AllM.(N{mc}).t)

    A.Summary           = AllM.(N{mc}).Overall;
    A.Regression        = AllM.(N{mc});

    close all; figure;
    clear PMset
    PMset.plottype      = 'box';
    PMset.UseT          = 't';%plot t or b values
    PMset.PlotSingle    = 0;
    PMset.savename      = ModN;
    PMset.plotpath      = SavePaths{m2r};
    PMset.Confidence    = 0.9999;
    PMset.AddAnalysis   = 1;%if set, this gives you addtional output to check the model
    PMset.ModPval       = size(fieldnames(A.Summary),1); % This simply multiplies p values (e.g., set to number of regressors for Bonferroni correction)
    AGFHK_Plot_Model( A, [], [], PMset )
end
