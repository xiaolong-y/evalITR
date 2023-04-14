#' Evaluate ITR
#'
#' @param outcome Outcome variable (or a list of outcome variables). Only takes in numeric values for both continous outcomes and binary outcomes (0 or 1).
#' @param treatment Treatment variable
#' @param data
#'   A data frame that contains \code{outcome} and \code{treatment}.
#' @param algorithms
#'   List of machine learning algorithms.
#' @param plim
#'   Proportion of treated units.
#' @param n_folds
#'   Number of cross-validation folds. Default is 5.
#' @param covariates
#'   Covariates included in the model.
#' @param ratio
#'   Split ratio between train and test set under sample splitting. Default is 0.
#' @param ngates
#'   The number of groups to separate the data into. The groups are determined by tau. Default is 5.
#' @import dplyr
#' @importFrom rlang !! sym
#' @export
#' @return An object of \code{itr} class
run_itr <- function(
    outcome,
    treatment,
    covariates,
    data,
    algorithms,
    plim,
    n_folds = 5,
    ratio = 0,
    ngates = 5,
    quantities = c("PAPE", "PAPDp", "AUPEC", "GATE")
) {


  ## number of algorithms
  n_alg <- length(algorithms)


  ## some working variables
  n_df <- nrow(data)
  n_X  <- length(data) - 1
  NFOLDS <- n_folds

  params <- list(
    n_df = n_df, n_folds = n_folds, n_alg = n_alg, ratio = ratio, ngates = ngates
  )


  ## loop over all outcomes
  estimates <- qoi <- vector("list", length = length(outcome))
  for (m in 1:length(outcome)) {

    ## data to use
    ## rename outcome and treatment variable
    data_filtered <- data %>%
      select(Y = !!sym(outcome[m]), Treat = !!sym(treatment), all_of(covariates))


    # if (n_folds == 0) {
    if (ratio > 0) {
    # might be a better approach; now if ratio > 0, then sp regardless folds
    # if (ratio = 0) {
    #   params$n_folds <- params$n_folds

      params$n_folds <- 0

      ## run under sample splitting
      estimates[[m]] <- itr_single_outcome(
        data       = data_filtered,
        algorithms = algorithms,
        params     = params,
        folds      = folds,
        plim       = plim
      )

      ## format output
      qoi[[m]] <- compute_qoi(estimates[[m]], algorithms, cv = FALSE, quantities)

    } else {

      # set ratio default value as 0 under cross validation
      params$ratio <- 0

      ## create folds
      treatment_vec <- data_filtered %>% dplyr::pull(Treat)
      folds <- caret::createFolds(treatment_vec, k = NFOLDS)

      ## run under cross validation
      estimates[[m]] <- itr_single_outcome(
        data       = data_filtered,
        algorithms = algorithms,
        params     = params,
        folds      = folds,
        plim       = plim
      )

      ## format output
      qoi[[m]] <- compute_qoi(estimates[[m]], algorithms, cv = TRUE, quantities)

    }
  }

  out <- list(qoi       = qoi,
              estimates = estimates)

  class(out) <- c("itr", class(out))

  return(out)

}


#' Evaluate ITR for Single Outcome
#'
#' @importFrom purrr map
#' @importFrom dplyr pull
#' @param data A dataset.
#' @param algorithms Machine learning algorithms.
#' @param params A list of parameters.
#' @param folds Number of folds.
#' @param plim The maximum percentage of population that can be treated under the budget constraint.

itr_single_outcome <- function(
    data,
    algorithms,
    params,
    folds,
    plim
) {

  ## obj to store outputs
  fit_ml <- lapply(1:params$n_alg, function(x) vector("list", length = params$n_folds))
  names(fit_ml) <- algorithms

  Tcv <- dplyr::pull(data, "Treat")
  Ycv <- dplyr::pull(data, "Y")
  indcv <- rep(0, length(Ycv))


  params$n_tb <- max(table(indcv))

  if (params$n_folds == 0) {

    ## ---------------------------------
    ## sample split
    ## ---------------------------------

    # create split series of test/training partitions
    split <- caret::createDataPartition(data$Treat,
                                        p = params$ratio,
                                        list = FALSE)
    trainset = data[split,]
    testset = data[-split,]

    ## ---------------------------------
    ## run ML
    ## ---------------------------------

    # prepare data
    training_data_elements <- create_ml_arguments(
      outcome = "Y", treatment = "Treat", data = trainset
    )

    testing_data_elements <- create_ml_arguments(
      outcome = "Y", treatment = "Treat", data = testset
    )

    total_data_elements <- create_ml_arguments(
      outcome = "Y", treatment = "Treat", data = data
    )

    ##
    ## run each ML algorithm
    ##
    if ("causal_forest" %in% algorithms) {
      fit_ml[["causal_forest"]] <- run_causal_forest(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        plim      = plim,
        indcv     = 1, #indcv and iter set to 1 for sample splitting
        iter      = 1
      )
    }

    if("lasso" %in% algorithms){
      fit_ml[["lasso"]] <- run_lasso(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("svm" %in% algorithms){
      fit_ml[["svm"]] <- run_svm(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }


    if("bartc" %in% algorithms){
      fit_ml[["bartc"]] <- run_bartc(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("bart" %in% algorithms){
      fit_ml[["bart"]] <- run_bartmachine(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("boost" %in% algorithms){
      fit_ml[["boost"]] <- run_boost(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("random_forest" %in% algorithms){
      fit_ml[["random_forest"]] <- run_random_forest(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("bagging" %in% algorithms){
      fit_ml[["bagging"]] <- run_bagging(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
    }

    if("cart" %in% algorithms){
      fit_ml[["cart"]] <- run_cart(
        dat_train = training_data_elements,
        dat_test  = testing_data_elements,
        dat_total = total_data_elements,
        params    = params,
        indcv     = 1,
        iter      = 1,
        plim      = plim
      )
}
  } else {

    ratio <- 0
    ## loop over j number of folds

    for (j in seq_len(params$n_folds)) {

      ## ---------------------------------
      ## data split
      ## ---------------------------------
      testset  <- data[folds[[j]], ]
      trainset <- data[-folds[[j]], ]
      indcv[folds[[j]]] <- rep(j, nrow(testset))


      ## ---------------------------------
      ## run ML
      ## ---------------------------------

      ## prepare data
      training_data_elements <- create_ml_arguments(
        outcome = "Y", treatment = "Treat", data = trainset
      )

      testing_data_elements <- create_ml_arguments(
        outcome = "Y", treatment = "Treat", data = testset
      )

      total_data_elements <- create_ml_arguments(
        outcome = "Y", treatment = "Treat", data = data
      )


      ##
      ## run each ML algorithm
      ##
      if ("causal_forest" %in% algorithms) {
        fit_ml[["causal_forest"]][[j]] <- run_causal_forest(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("lasso" %in% algorithms){
        fit_ml[["lasso"]][[j]] <- run_lasso(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("svm" %in% algorithms){
        fit_ml[["svm"]][[j]] <- run_svm(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }


      if("bartc" %in% algorithms){
        fit_ml[["bartc"]][[j]] <- run_bartc(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("bart" %in% algorithms){
        fit_ml[["bart"]][[j]] <- run_bartmachine(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("boost" %in% algorithms){
        fit_ml[["boost"]][[j]] <- run_boost(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("random_forest" %in% algorithms){
        fit_ml[["random_forest"]][[j]] <- run_random_forest(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("bagging" %in% algorithms){
        fit_ml[["bagging"]][[j]] <- run_bagging(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

      if("cart" %in% algorithms){
        fit_ml[["cart"]][[j]] <- run_cart(
          dat_train = training_data_elements,
          dat_test  = testing_data_elements,
          dat_total = total_data_elements,
          params    = params,
          indcv     = indcv,
          iter      = j,
          plim      = plim
        )
      }

    } ## end of fold

  }

  return(list(
    params = params, fit_ml = fit_ml,
    Ycv = Ycv, Tcv = Tcv, indcv = indcv, plim = plim
  ))
}

#' Estimate quantity of interests
#' @param fit Fitted model. Usually an output from \code{estimate_itr}
#' @param ... Further arguments passed to the function.
#' @return An object of \code{itr} class
#' @export
evaluate_itr <- function(fit, ...){

  estimates  <- fit$estimates
  cv         <- estimates[[1]]$params$cv
  df         <- fit$df
  algorithms <- fit$df$algorithms
  outcome    <- fit$df$outcome

  qoi        <- vector("list", length = length(outcome))

  ## loop over all outcomes
  for (m in 1:length(outcome)) {

    ## compute qoi
    qoi[[m]] <- compute_qoi(estimates[[m]], algorithms)

  }

  out <- list(
    qoi = qoi, cv = cv, df = df, estimates = estimates)

  class(out) <- c("itr", class(out))

  return(out)

}

#' Conduct hypothesis tests
#' @param fit Fitted model. Usually an output from \code{estimate_itr}
#' @param ngates The number of groups to separate the data into. The groups are determined by \code{tau}. Default is 5.
#' @param nsim Number of Monte Carlo simulations used to simulate the null distributions. Default is 10000.
#' @param ... Further arguments passed to the function.
#' @return An object of \code{itr} class
#' @export
test_itr <- function(
    fit,
    nsim = 10000,
    ...
) {

  # test parameters
  estimates  <- fit$estimates
  cv         <- estimates[[1]]$params$cv
  fit_ml     <- estimates[[1]]$fit_ml
  Tcv        <- estimates[[1]]$Tcv
  Ycv        <- estimates[[1]]$Ycv
  indcv      <- estimates[[1]]$indcv
  n_folds    <- estimates[[1]]$params$n_folds
  ngates     <- estimates[[1]]$params$ngates
  algorithms <- fit$df$algorithms
  outcome    <- fit$df$outcome
  # run tests

  ## =================================
  ## sample splitting
  ## =================================

  if(cv == FALSE){
    cat('Conduct hypothesis tests for GATEs unde sample splitting ...\n')

    ## create empty lists to for consistcv and hetcv
    consist <- list()
    het <- list()

      # model with a single outcome
      ## run consistency and heterogeneity tests for each model
      if ("causal_forest" %in% algorithms) {
        consist[["causal_forest"]] <- consist.test(
          T   = Tcv,
          tau = fit_ml$causal_forest$tau,
          Y   = Ycv,
          ngates = ngates)

        het[["causal_forest"]] <- het.test(
          T   = Tcv,
          tau = fit_ml$causal_forest$tau,
          Y   = Ycv,
          ngates = ngates)
      }

      if ("lasso" %in% algorithms) {
        consist[["lasso"]] <- consist.test(
          T   = Tcv,
          tau = fit_ml$lasso$tau,
          Y   = Ycv,
          ngates = ngates)

        het[["lasso"]] <- het.test(
          T   = Tcv,
          tau = fit_ml$lasso$tau,
          Y   = Ycv,
          ngates = ngates)
      }

    if ("svm" %in% algorithms) {
      consist[["svm"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$svm$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["svm"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$svm$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("bartc" %in% algorithms) {
      consist[["bartc"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$bartc$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["bartc"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$bartc$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("bart" %in% algorithms) {
      consist[["bart"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$bart$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["bart"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$bart$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("boost" %in% algorithms) {
      consist[["boost"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$boost$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["boost"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$boost$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("random_forest" %in% algorithms) {
      consist[["random_forest"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$random_forest$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["random_forest"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$random_forest$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("bagging" %in% algorithms) {
      consist[["bagging"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$bagging$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["bagging"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$bagging$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("cart" %in% algorithms) {
      consist[["cart"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$cart$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["cart"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$cart$tau,
        Y   = Ycv,
        ngates = ngates)
    }

    if ("caret" %in% algorithms) {
      consist[["caret"]] <- consist.test(
        T   = Tcv,
        tau = fit_ml$caret$tau,
        Y   = Ycv,
        ngates = ngates)

      het[["caret"]] <- het.test(
        T   = Tcv,
        tau = fit_ml$caret$tau,
        Y   = Ycv,
        ngates = ngates)
    }

  }

  ## =================================
  ## cross validation
  ## =================================

  if(cv == TRUE){
    cat('Conduct hypothesis tests for GATEs unde cross-validation ...\n')

      ## create empty lists to for consistcv and hetcv
    consistcv <- list()
    hetcv <- list()

      ## run consistency and heterogeneity tests for each model
      if ("causal_forest" %in% algorithms) {
        consistcv[["causal_forest"]] <- consistcv.test(
          T   = Tcv,
          tau = gettaucv(fit),
          Y   = Ycv,
          ind = indcv,
          ngates = ngates)

        hetcv[["causal_forest"]] <- hetcv.test(
          T   = Tcv,
          tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
          Y   = Ycv,
          ind = indcv,
          ngates = ngates)
      }

    # if ("lasso" %in% algorithms) {
    #   consistcv[["lasso"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["lasso"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("bartc" %in% algorithms) {
    #   consistcv[["bartc"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["bartc"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("bart" %in% algorithms) {
    #   consistcv[["bart"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["bart"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("boost" %in% algorithms) {
    #   consistcv[["boost"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["boost"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("random_forest" %in% algorithms) {
    #   consistcv[["random_forest"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["random_forest"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("bagging" %in% algorithms) {
    #   consistcv[["bagging"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["bagging"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("cart" %in% algorithms) {
    #   consistcv[["cart"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit),
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["cart"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = gettaucv(fit), # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
    #
    # if ("caret" %in% algorithms) {
    #   consistcv[["caret"]] <- consistcv.test(
    #     T   = Tcv,
    #     tau = fit_ml$caret$tau_cv,
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    #
    #   hetcv[["caret"]] <- hetcv.test(
    #     T   = Tcv,
    #     tau = fit_ml$caret$tau_cv, # a matrix of length(total sample size) x n_folds
    #     Y   = Ycv,
    #     ind = indcv,
    #     ngates = ngates)
    # }
  }

  # formulate and return output
  if(cv == FALSE){
    tests <- list(consist = consist,
                  het = het)
    return(tests)
  }
  if(cv == TRUE){
    tests_cv <- list(consistcv = consistcv,
                    hetcv = hetcv)
    return(tests_cv)
  }

}


utils::globalVariables(c("Treat", "aupec", "sd", "Pval", "aupec.y", "fraction", "AUPECmin", "AUPECmax", ".", "fit", "out", "pape", "alg", "papep", "papd", "type", "gate", "group", "qnorm", "vec"))
