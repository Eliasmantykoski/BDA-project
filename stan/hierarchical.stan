data {
  int<lower=0> N; // Number of age groups
  int<lower=0> Y; // Number of years
  matrix[N, Y] accidentData; // Accident rate data for age groups over years
  int xpred; // Year for prediction (e.g., 2000)
}

parameters {
  // Group-level parameters
  real mu_alpha;            // Mean intercept
  real mu_beta;             // Mean slope
  real<lower=0> sigma_alpha; // SD of intercepts
  real<lower=0> sigma_beta;  // SD of slopes
  real<lower=0> sigma;       // Observation noise
  
  // Varying intercepts and slopes for each age group
  vector[N] alpha;          // Intercepts for age groups
  vector[N] beta;           // Slopes for age groups
}

transformed parameters {
  matrix[N, Y] mu;          // Mean accident rates for all groups and years
  for (i in 1:N)
    for (j in 1:Y)
      mu[i, j] = alpha[i] + beta[i] * (j - 1); // Time starts at year 1
}

model {
  // Priors for group-level parameters
  mu_alpha ~ normal(0, 10);     // Prior for mean intercept
  mu_beta ~ normal(0, 5);       // Prior for mean slope
  sigma_alpha ~ cauchy(0, 2);   // Prior for SD of intercepts
  sigma_beta ~ cauchy(0, 2);    // Prior for SD of slopes
  sigma ~ cauchy(0, 2);         // Prior for observation noise

  // Priors for varying intercepts and slopes (hierarchical structure)
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);

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
    pred[i] = normal_rng(alpha[i] + beta[i] * (xpred - 1999), sigma);
    
    // Generate replicated data
    for (j in 1:Y)
      yrep[i, j] = normal_rng(mu[i, j], sigma);
  }
}