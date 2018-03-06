mu1 = [1.18 1.84];
mu2 = [1.15 1.93];
mu3 = [1.15 1.90];
mu4 = [1.17 1.86];
mu5 = [1.17 1.91];
Sigma1 = [1 0.2;0.2 1.5];
Sigma2 = [1.9 -0.17;-0.17 0.97];
Sigma3 = [0.7 -0.26;-0.26 0.7];
Sigma4 = [1.7 -0.38;-0.38	0.7];
Sigma5 = [0.4 -1.3;-1.3	5.7];
x1 = -2:0.05:4; x2 = 0:0.05:4;
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],mu1,Sigma1);
F1 = reshape(F1,length(x2),length(x1));
F2 = mvnpdf([X1(:) X2(:)],mu2,Sigma2);
F2 = reshape(F2,length(x2),length(x1));
F3 = mvnpdf([X1(:) X2(:)],mu3,Sigma3);
F3 = reshape(F3,length(x2),length(x1));
F4 = mvnpdf([X1(:) X2(:)],mu4,Sigma4);
F4 = reshape(F4,length(x2),length(x1));
F5 = mvnpdf([X1(:) X2(:)],mu5,Sigma5);
F5 = reshape(F5,length(x2),length(x1));
