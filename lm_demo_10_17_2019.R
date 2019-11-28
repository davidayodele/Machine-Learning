### practice with a toy data example

library(dplyr)
library(ggplot2)

### create the real inputs with a MVN
mu_vector <- c(0, 0)
sigma_1 <- 0.7
sigma_2 <- 1.2
rho_12 <- 0.25

covmat_x <- matrix(c(sigma_1^2, 
                     rho_12*sigma_1*sigma_2,
                     rho_12*sigma_1*sigma_2,
                     sigma_2^2),
                   nrow = 2,
                   byrow = 2)

covmat_x

set.seed(12345)
Xreal <- MASS::mvrnorm(n = 300,
                       mu = mu_vector,
                       Sigma = covmat_x)

Xreal %>% head()

Xreal %>% dim()

Xreal %>% class()

Xreal <- Xreal %>% as.data.frame() %>% tbl_df()

Xreal

### create the fake or inactive inputs
set.seed(34234)
Xfake <- MASS::mvrnorm(n = nrow(Xreal),
                       mu = rep(0, 10),
                       Sigma = diag(10)) %>% 
  as.data.frame() %>% tbl_df()

Xfake

### assemble all inputs into a common tibble
Xinputs <- Xreal %>% bind_cols(Xfake)

Xinputs <- Xinputs %>% 
  purrr::set_names(sprintf("x%02d", 1:ncol(Xinputs)))

Xinputs %>% names()

### define the true relationship
my_true_func <- function(x1, x2, beta_vec)
{
  beta_vec[1] + beta_vec[2] * x1 + beta_vec[3] * x2 +
    beta_vec[4] * x1 * x2
}

### calculate the true linear predictor
beta_true <- c(-0.33, 2, 1, -1.7)

true_data <- Xinputs %>% 
  mutate(mu = my_true_func(x01, x02, beta_true))

true_data

### visualize mu wrt x1 and x2
true_data %>% 
  select(x01, x02, mu) %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name",
                value = "input_value",
                -obs_id, -mu) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = mu)) +
  geom_point(alpha = 0.5) +
  facet_grid( ~ input_name) + 
  theme_bw()

### create noisy observations around the truth
sigma_true <- 1
set.seed(83032)
all_data <- true_data %>% 
  mutate(y = rnorm(n = n(),
                   mean = mu,
                   sd = sigma_true))

### look at the noisy response wrt all inputs
all_data %>% 
  select(y, starts_with("x")) %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name", 
                value = "input_value",
                -obs_id, -y) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = y)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~input_name) +
  theme_bw()

### data split

set.seed(348713)
train_id <- sample(1:nrow(all_data), 100)

train_df <- all_data %>% 
  select(-mu) %>% 
  slice(train_id)

holdout_df <- all_data %>% 
  select(-mu) %>% 
  slice(-train_id)

### visualize the training set
train_df %>% 
  select(y, starts_with("x")) %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name", 
                value = "input_value",
                -obs_id, -y) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = y)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~input_name) +
  theme_bw()

### fit a simple linear model with `lm()`

mod1 <- lm(y ~ x01, train_df)

summary(mod1)

### install.package("coefplot")

coefplot::coefplot(mod1)

### use three inputs
mod3 <- lm(y ~ x01 + x02 + x03, train_df)

coefplot::coefplot(mod3)

summary(mod3)

### fit with all inputs additive
mod_all <- lm(y ~ ., train_df)

coefplot::coefplot(mod_all)

coefplot::multiplot(mod1, mod3, mod_all)

### create the model with the interaction term
mod_true <- lm(y ~ x01*x02, train_df)

summary(mod_true)

coefplot::multiplot(mod1, mod3, mod_true)


### quadratic model
mod_quad <- lm( y ~ x01 + x02 + I(x02^2), train_df)

coefplot::coefplot(mod_quad)

summary(mod_quad)

### how many columns are in all model with ALL PAIR-wise interactions

dim(model.matrix(y ~ (.)^2, train_df))

model.matrix( y ~ (.)^2, train_df) %>% colnames()

### build the model with all pairwise interactions
mod_big <- lm(y ~ (.)^2, train_df)

summary(mod_big)

coefplot::coefplot(mod_big)

### a quadratic model with all pairwise interactions and quadratic terms

Xmat <- Xinputs %>% as.matrix()

design_matrix_all_quad <- model.matrix( ~ poly(Xmat, degree = 2, raw = TRUE))

dim(design_matrix_all_quad)
