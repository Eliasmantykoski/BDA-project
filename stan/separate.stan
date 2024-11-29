data {
  int<lower=0> N; // Number of age groups
  int<lower=0> Y; // Number of years
  matrix[N, Y] accidentData; // Accident rate data for age groups over years
  int xpred; // Year for prediction (e.g., 2000)
}

parameters {
  vector[N] alpha; // Intercept for each age group
  vector[N] beta;  // Trend (slope) for each age group
  vector<lower=0>[N] sigma; // Standard deviation for each age group
}

transformed parameters {
  matrix[N, Y] mu; // Mean accident rates for each age group and year
  for (i in 1:N)
    for (j in 1:Y)
      mu[i, j] = alpha[i] + beta[i] * (j - 1); // Time starts at year 1
}

model {
  // Priors for alpha, beta, and sigma
  alpha ~ normal(1.5, 10);       // Prior for intercepts
  beta ~ normal(0, 5);         // Prior for slopes
  sigma ~ cauchy(0, 2);        // Prior for standard deviations

  // Likelihood for the observed data
  for (i in 1:N)
    for (j in 1:Y)
      accidentData[i, j] ~ normal(mu[i, j], sigma[i]);
}

generated quantities {
  matrix[N, Y] yrep;           // Posterior predictive replicated data
  vector[N] pred;              // Predictions for the specified year `xpred`
  matrix[N, Y] log_lik;
  for (i in 1:N) {
    // Predict accident rates for year `xpred`
    pred[i] = normal_rng(alpha[i] + beta[i] * (xpred - 1999), sigma[i]);
    
    // Generate replicated data
    for (j in 1:Y){
      yrep[i, j] = normal_rng(mu[i, j], sigma[i]);
      log_lik[i, j] = normal_lpdf(accidentData[i, j] | mu[i, j], sigma[i]);
      }
  }
}