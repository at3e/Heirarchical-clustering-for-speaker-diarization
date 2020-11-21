function mog = init_gm(x,K)
dim = size(x, 2);
Vars = zeros(K, dim, dim);
[Cidx,C]=kmeans(x,K);
for k=1:K
    Clus=[];
    for m=1:length(Cidx)
        if Cidx(m)==k
            Clus = [Clus;x(m,:)];
        end
    end
    Cvar = diag(nancov(Clus));
    invCvar = diag(1./Cvar);
    Vars(k, :, :) = invCvar;
end
    
gm.mean = C; 
gm.covinv = Vars;
gm.prior = repmat(1/k,k,1);
return
