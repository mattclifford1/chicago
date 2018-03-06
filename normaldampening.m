mu = [0 0];
Sigma = [1 0.5; 0.5 1];
x1 = -5:0.1:5; x2 = -5:0.1:5;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-5 5 -5 5 0 .2])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
Fdisp = 0;
h = figure;
axis tight manual
filename = 'testAnimated.gif';
for i=1:100
    Fdamp = mvnpdf([X1(:) X2(:)],[0 0],[1 0.5; 0.5 1]);
    Fdamp = reshape(Fdamp,length(x2),length(x1));
    F = F - 0.005*Fdamp;
    F(F<0) = 0;
    if rand(1)>0.975
        Fdisp = mvnpdf([X1(:) X2(:)],[5*rand(1) 5*rand(1)],[0.15 0.05; 0.05 0.15]);
        Fdisp = reshape(Fdisp,length(x2),length(x1));
    end  
    F = F + 0.001*Fdisp;
    surf(x1,x2,F)
    xaxis = xlabel('X-Coordinate');
    yaxis = ylabel('Y-Coordinate');
    zaxis = zlabel('Probability of Crime');
    set(xaxis, 'FontSize', 10);
    set(yaxis, 'FontSize', 10);
    set(zaxis, 'FontSize', 10);
    colorbar;
    axis([-5 5 -5 5 0 .2])
    drawnow
          % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if i == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end