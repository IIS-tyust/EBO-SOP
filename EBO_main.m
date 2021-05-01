function EBO_main()
%------------------Bayesian Evolutionary Optimization----------------------
clear
clc
addpath(genpath(pwd));
%Parameter setting
fucnum = 1; %Ellipsoid function
D = 10;
FEmax = 11*D;
gmax = 20;
theta = 5.*ones(1,D);

%% Initialization
lu = Initialize(fucnum,D);
Lb = lu(1,:);
Ub = lu(2,:);
Boundary = [Ub;Lb];
%Initialize the training archive
Nt = 2*D;
Alhs = lhsdesign(Nt,D);
Arc = repmat(Lb,Nt,1) + Alhs.* repmat((Ub-Lb),Nt,1);
Arc = unique(Arc,'rows');
Arc(:,D+1) = fitness(Arc, fucnum);
FEnum = size(Arc,1);

%Optimization
while FEnum < FEmax
    ArcDec = Arc(:,1:D);
    ArcObj = Arc(:,end);
    %Train a GP model
    GPmodel = dacefit(ArcDec,ArcObj,'regpoly0','corrgauss',theta,1e-5.*ones(1,D),100.*ones(1,D));
    %Generate an offspring
    %Initialize a parent population
    Np = 50;
    Plhs = lhsdesign(Np,D);
    Parent = repmat(Lb,Np,1) + Plhs.* repmat((Ub-Lb),Np,1);
    g = 0;
    EIbest = 0;
    while g < gmax
        g = g+1;
        Offspring = GA(Parent,Boundary);
        Popcom = [Parent;Offspring];
        N = size(Popcom,1);
        Abest = min(Arc(:,end));
        for i = 1: N
            [PopObj(i,1),~,MSE(i,1)] = predictor(Popcom(i,:),GPmodel); 
            s(i,1) = sqrt(MSE(i,1));
            lamda(i,1) = (Abest-PopObj(i,1))./s(i,1);
            EI(i,1) = (Abest-PopObj(i,1)).* Gaussian_CDF(lamda(i,1)) + s(i,1)*Gaussian_PDF(lamda(i,1));
        end
        [EIsort,index] = sort(EI,'descend');
        Parent = Popcom(index(1:Np),:);
        
        if EIbest < EI(index(1))
            EIbest = EI(index(1));
            best = Popcom(index(1),:);
        end
    end
    
    best(:,D+1) = fitness(best, fucnum);
    FEnum = FEnum + size(best,1)
    %Update the training archive
    Arctemp = [Arc; best];
    [~,ind] = unique(Arctemp(:,1:D),'rows');
    Arc     = Arctemp(ind,:);
    Pbest   = min(Arc(:,end))
    plot(FEnum,Pbest,'-o');
    hold on
    getframe;
end
return