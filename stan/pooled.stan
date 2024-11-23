data {
  int<lower=0> N; // age group
  int<lower=0> Y; // year
  matrix[N,Y] accidentData;//accident data
  int xpred;
}


parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

transformed parameters{
  vector[Y]mu;
  //linear model
  for(j in 1:Y)
    mu[j]=alpha+beta*j;
}


model {

    alpha~normal(0,100);
    beta~normal(0,10);

  for(j in 1:Y){
    accidentData[,j]~normal(mu[j],sigma);
  }
}

generated quantities{
  //log likelihood
  matrix[N,Y] log_lik;
  matrix[N,Y] yrep;
  //accident prediction in 2020 in different police force
  vector[N] pred;
  for(i in 1:N){
    // 2005 -> 1, 2006 -> 2, ..., 2020 -> 16 
    pred[i]=normal_rng(alpha+beta*(xpred-2004),sigma);
  }
  
  for(i in 1:N){
    for(j in 1:Y){
      // do posterior sampling and try to reproduce the original data
      yrep[i,j]=normal_rng(mu[j],sigma);
      // prepare log likelihood for PSIS-LOO 
      log_lik[i,j]=normal_lpdf(accidentData[i,j]|mu[j],sigma);
    }
  }
  
}