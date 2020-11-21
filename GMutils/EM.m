function gm = EM(x,gm,stopcrit,reg)
  [n,dim] = size(x);
  k = length(gm.prior);
  iter = 1;
  LL2 = -inf;
  
  while (iter<=stopcrit.maxiters)

	% estimate responsibilities
	[p,p_k] = gm_prob(x,gm);

	% check for improvement in likelihood
	LL1 = LL2;
	LL2 = sum(log(p));
	if ((LL2-LL1)/abs(LL1))<stopcrit.minllimpr break; end

	% normalize:
	p(p==0) = 1; % avoid dividing by zero
	p_k = bsxfun(@rdivide,p_k,p);

	% update parameters
	new_priors = sum(p_k,1)';
	new_priors(new_priors==0) = 10*realmin;
	new_mean = p_k' * x;

	gm.prior = new_priors/n;
	gm.mean = bsxfun(@rdivide, new_mean, new_priors);
	for i=1:k
		df = bsxfun(@times, bsxfun(@minus,x,gm.mean(i,:)),sqrt(p_k(:,i)));
		gm.covinv(i,:,:) = inv((df'*df)/new_priors(i) + reg*eye(dim));
	end

	iter = iter+1;
  end

end
