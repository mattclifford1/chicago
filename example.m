par.l = 1;
par.deltat = 1/100;
par.omega = 1/15;
par.eta = 0.03;
par.Theta = 0.56;
par.Gamma = 0.019*10;

A0 = ones(128, 1)/30;
B0 = ones(128,1)*par.Theta*par.Gamma/par.omega;
C0 = zeros(128,1);

% Simulate for 365 days
[B, C] = crime1d([par.deltat, par.Gamma, par.omega, par.eta, par.l, par.Theta], A0, B0, C0, 100*365, 1);
