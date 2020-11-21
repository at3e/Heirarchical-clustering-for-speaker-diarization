function [p,p_k] = gm_prob(x,gm)
  K = size(gm.mean,1);
  [n,dim] = size(x);
  p_k = zeros(n,K);

  for i=1:K
    xm = x - repmat(gm.mean(i,:),n,1);
    C = squeeze(gm.coninv(i,:,:));
    d2 = sum((xm*C).*xm,2);
    p_k(:,i) = gm.prior(i)*exp(-d2/2)*sqrt(det(C)/(2*pi)^dim);
  end
  p = sum(p_k,2);

end