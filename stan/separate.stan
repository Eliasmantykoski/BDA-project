data {
  int<lower=0> N; // the number of age groups
  int<lower=0> Y; // the number of years has been studied, year 2000 corresponds to 1
  matrix[N,Y] accidentData;//accident data
  int xpred;
}

parameters {
  vector[N] alpha;
  vector[N] beta;
  vector<lower=0>[N] sigma;
}

transformed parameters{
  matrix[N,Y]mu;
  for(i in 1:N)
    for(j in 1:Y)
      mu[i,j]=alpha[i]+beta[i]*j;
}

model {
   for(i in 1:N){
       alpha[i]~normal(300,100);
       beta[i]~normal(0,10);
    }

  //for each age group
  for(i in 1:N){
    //for each observed year
    for(j in 1:Y){
     accidentData[i,j]~normal(mu[i,j],sigma[i]);
    }
  }
}


generated quantities{
  //log likelihood
  matrix[N,Y] log_lik;
  matrix[N,Y] yrep;

  vector[N] pred;
  
  for(i in 1:N){
    pred[i]=normal_rng(alpha[i]+beta[i]*(xpred-1999),sigma[i]);
  }
  
  for(i in 1:N){
    for(j in 1:Y){
      // do posterior sampling and try to reproduce the original data
      yrep[i,j]=normal_rng(mu[i,j],sigma[i]); 
      // prepare log likelihood for PSIS-LOO 
      log_lik[i,j]=normal_lpdf(accidentData[i,j]|mu[i,j],sigma[i]);
    }
  }
}