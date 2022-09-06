
<!-- README.md is generated from README.Rmd. Please edit that file -->

# evalITR

<!-- badges: start -->
<!-- badges: end -->

## Installation

You can install the development version of evalITR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("MichaelLLi/evalITR")
```

(Optional) if you have multiple cores, we recommendate using multisession futures and processing in parallel. This would increase computation efficiency and reduce the time to fit the model. 

```r
library(furrr)
library(future.apply)

nworkers <- 4
plan(multisession, workers =nworkers)
```

## Example

This is an example using the `star` dataset (for more information about the dataset, please use `?star`). 

We first load the dataset and specify both outcome variables (reading, math, and writing scores) and
covariates we want to include in the model. Then we use a series of machine learning
algorithms to estimate the heterogeneous effects of small classes on
educational attainment. We use 20% as a budget constraint and tuned the
model through through the 3-fold cross validation.

``` r
library(tidyverse)
library(evalITR)

load("data/star.rda")

# specifying outcomes
outcomes <- c("g3tlangss",
                "g3treadss","g3tmathss")

# specifying covariates
covariates <-  star %>% 
                dplyr::select(-c(all_of(outcomes),"treatment")) %>% 
                colnames()

# estimate ITR 
fit <- run_itr(outcome = outcomes,
               treatment = "treatment",
               covariates = covariates,
               data = star,
               algorithms = c(
                  "causal_forest", 
                  "bart",
                  "lasso",
                  "boost", 
                  "random_forest",
                  "bagging",
                  "cart"),
               plim = 0.2,
               n_folds = 3)
```

We plot the estimated Area Under the Prescriptive Effect Curve (AUPEC)
for the writing score across a range of budget constraints for different algorithms.

``` r
# plot the AUPEC with different ML algorithms
plot_aupec(fit = fit$qoi[1], 
          outcome = outcomes[1],
          treatment = "treatment",
          data = star, 
          algorithms = c(
                  "causal_forest",
                  "bart",
                  "lasso",
                  "boost", 
                  "random_forest",
                  "bagging", 
                  "cart")) 
```

![](man/figures/README-plot-1.png)<!-- -->
