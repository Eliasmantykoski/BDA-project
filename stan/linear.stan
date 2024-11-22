data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  // priors
  alpha ~ normal(42000, 1000);
  beta ~ normal(0, 100);
  sigma ~ cauchy(0, 2.5); 
  
  y ~ normal(alpha + beta * x, sigma);
}