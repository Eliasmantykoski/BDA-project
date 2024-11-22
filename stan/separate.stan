data {
  int<lower=0> N; // the number of age groups
  int<lower=0> Y; // the number of years has been studied, year 2000 corresponds to 1
  matrix[N,Y] accidentData;//accident data
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
