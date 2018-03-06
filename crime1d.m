function [B, C] = crime1d(par, A0, B, C, steps, doplot)
% function [B, C] = crime1d(par, A0, B, C, steps, doplot)

% Number of times steps to take
if ~exist('steps', 'var')
    steps = 1;
end
% Whether or not to plot
if ~exist('doplot', 'var')
    doplot = 0;
end

% Size of the lattice
n = numel(A0);

% Copy the parameters for ease of use
deltat = par(1);
Gamma = par(2);
omega = par(3);
eta = par(4);
l = par(5);
Theta = par(6);

% Make sure that everything has the correct shape
sizeB = size(B);
sizeC = size(C);
A0 = A0(:);
B = B(:);
C = C(:);

% Get the adjacent nodes' indices
idx = (1:n)';
adj1 = [idx(2:end); idx(1)];
adj2 = [idx(end); idx(1:end-1)];
adj = [adj1(:), adj2(:)];

% Plotting handles
if doplot
    sc = max(max(B))/20;
    figure();
    subplot(1, 2, 1);
    imC = image(C'*128);
    txtT = text(0, 0.45, 'Time:');
    txtC = text(n*2/3, 0.45, 'Criminals:');
    subplot(1, 2, 2);
    imB = image(B'/sc);
    txtE = text(n*1/3, 0.45, 'Crimes:');
    colormap(jet);
end
    
% Iterate in time
for t = 1:steps
    
    % Create burglaries matrix (integer)
    E = zeros(n, 1);

    % Calculate overall attractivity
    A = A0 + B;
    
    % Get the adjacent values of A
    Aadj1 = [A(2:end); A(1)];
    Aadj2 = [A(end); A(1:end-1)];
    Aadj = [Aadj1(:), Aadj2(:)];

    % Copy of the criminals matrix
    newC = C;
    
    % Iterate over each site
    s = 1;
    while s <= n
        
        % Calculate the probability of burglary
        p = 1 - exp(-A(s)*deltat);
        
        % Calculate movement probabilities
        q = Aadj(s, :)/(Aadj(s, 1) + Aadj(s, 2)); % sum would be clean but it slows down a lot
        if (Aadj(s, 1) + Aadj(s, 2)) == 0
            q = [0.5, 0.5];
        end
            
        % Iterate over each burglar
        for i = 1:C(s)

            % Remove the burglar (always happens)
            newC(s) = newC(s) - 1;
            
            % Do they burgle?
            if rand() < p
                % Yes
                E(s) = E(s) + 1;
            else
                % No, move instead
                rnd = rand();
                j = 1;
                while rnd > q(j)
                    rnd = rnd - q(j);
                    j = j + 1;
                end
                % Add the burglar to a new site
                newC(adj(s, j)) = newC(adj(s, j)) + 1;
            end
            
        end
        
        % Create new burglars according to a Poisson process with rate Gamma
        rnd = rand();
        k = 0;
        p = exp(-Gamma*deltat); % Probability of zero created
        while rnd > p
            k = k + 1;
            p = p + exp(-Gamma*deltat)*(Gamma*deltat)^k/factorial(k);
        end
        newC(s) = newC(s) + k;
        
        % Consider next site
        s = s + 1;
        
    end
    
    % Update the criminals matrix
    C = newC;
    
    % Update the dynamic attractivity matrix
    DeltaB = (([B(2:end); B(1)] + [B(end); B(1:end-1)]) - 2*B)/l^2;
    B = (B + eta*l^2/2*DeltaB)*(1 - omega*deltat) + Theta*E;
    
    % Plotting
    if doplot
        set(imC, 'CData', C'*128);
        set(imB, 'CData', B'/sc);
        set(txtT, 'String', sprintf('Time: %.2f days', t*deltat));
        set(txtC, 'String', sprintf('Criminals: %d', sum(sum(C))));
        set(txtE, 'String', sprintf('Crimes: %d', sum(sum(E))));
        drawnow; 
    end        
end

B = reshape(B, sizeB);
C = reshape(C, sizeC);

end
