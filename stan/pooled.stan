data {
  int<lower=0> N; // Number of age groups
  int<lower=0> Y; // Number of years
  matrix[N, Y] accidentData; // Accident rate data for age groups over years
  int xpred; // Year for prediction (e.g., 2000)
}

parameters {
  real alpha;        // Global intercept
  real beta;         // Global slope
  real<lower=0> sigma; // Global noise
}

transformed parameters {
  matrix[N, Y] mu; // Mean accident rates for all groups and years
  for (i in 1:N)
    for (j in 1:Y)
      mu[i, j] = alpha + beta * (j - 1); // Time starts at year 1
}

model {
  // Priors for alpha, beta, and sigma
  alpha ~ normal(1.5, 10);    // Prior for the global intercept
  beta ~ normal(0, 5);      // Prior for the global slope
  sigma ~ cauchy(0, 2);     // Prior for the global standard deviation

  // Likelihood for the observed data
  for (i in 1:N)
    for (j in 1:Y)
      accidentData[i, j] ~ normal(mu[i, j], sigma);
}

generated quantities {
  matrix[N, Y] yrep;  // Posterior predictive replicated data
  vector[N] pred;     // Predictions for the specified year `xpred`
  for (i in 1:N) {
    // Predict accident rates for year `xpred`
    pred[i] = normal_rng(alpha + beta * (xpred - 1999), sigma);
    
    // Generate replicated data
    for (j in 1:Y)
      yrep[i, j] = normal_rng(mu[i, j], sigma);
  }
}