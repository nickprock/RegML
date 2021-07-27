# RegML 2021 Exam
## Nicola Procopio
## Report on the labs
In these labs we have focused on regularization methods and supervised learning models.
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.

$${(x_1, y_1), ..., (x_i, y_i), ... ,(x_n, y_n)}$$

### Lab 1: Binary classification and model selection
This lab addresses binary classification and model selection on synthetic data.
We start generating a sintetic dataset using *create_random_data* function, it's inside the *lab1_utils.py* script and generate a random dataset with *'linear'*. *'moons'* or *'circles'* shape, the shape is the relation between features and target variable. For the first part we generate a linear dataset with 100 samples and 2 features with noise level equals to 1.3.
We split the data in train set (80% of samples) and test set (20% of samples)
```
X, y = create_random_data(n_samples=100, noise_level=1.3, dataset="linear")
X_train, X_test, y_train, y_test = data_split(X, y, n_train=80)
```

![linear_dataset](linear_dataset_inserire)

#### Training a Linear Ridge-Regression Model
At the first classification we use **KernelRidge** function from *scikit-learn*. It uses the ridge regression and classification, in our case the ridge classification. Ridge penalize the size of the coefficients with the *L2-norm*, kernel ridge uses squared error loss combined with L2 regularization.

$$\min_{w} || X w - y||_2^2 + \alpha ||w||_2^2$$

The complexity parameter $\alpha \geq 0$ controls the amount of shrinkage, in our case $\alpha = 0.1$. We use a linear kernel but kernel ridge support also non-linear functions like RBF or polynomial.
```
regularization = 0.1

model = KernelRidge(regularization, kernel="linear")
model.fit(X_train, y_train)
```

The binary classification error is $$error=Avg(I(y_i, \hat{y_i}))$$ when the true and the predicted values are the same $I(.)$ return 0, 1 otherwise.
$$error = \frac{1}N \sum_{i=1}^N{I(y_i, \hat{y_i})}$$

```
def binary_classif_error(y_true, y_pred):
    return np.mean(np.sign(y_pred) != y_true)
```
Our error rate is 25% in both train and test as can be seen from the next figure.

![separation](separation)

We change the regularization parameter and plot the relation between regularization and error rate, a strong regularization increase the error rate. The lowest error rate is 25%, the dataset has few samples and the **linear kernel underfit** the data.

![reg_error](reg_error)

We encrease the number of data points and evaluate when change the test error for different number of points in train set, later we change the noise level in the dataset and observe how the error change. In both cases the regularization value and kernel are fixed.

```
# 2. Change in number of data-points
num_points = np.arange(5, 1000, 10)
np_test_errors = []
# TODO: Create the model
model = KernelRidge(alpha=0.1, kernel='linear')
for points in num_points:
    X, y = create_random_data(points + 20, 1, seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=points)
    # TODO: Fit/Predict
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    np_test_errors.append(binary_classif_error(y_test, test_preds))

# 3. Amount of noise in the data
data_noise = [0.3, 0.5, 1.0, 2.0]
noise_test_errors = []
# TODO: Fill the whole example in
model = KernelRidge(alpha=0.1, kernel='linear')
for noise in data_noise:
    X, y = create_random_data(1000, noise_level=noise, seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=800)
    # TODO: Fit/Predict
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    noise_test_errors.append(binary_classif_error(y_test, test_preds))
```

![data_error](data_error)

![noise_error](noise_error)

The changing in train - test percentage of samples impact randomly on error rate, on the other way the changing in noise and error are almost directly proportional.

To find the optimal value of $\alpha$ we use the cross validation with 5 folds:
* The regularization parameter with minimal error is 1.0e-07
* Achieving a 5-fold CV average error of 0.25%

#### Kernel Ridge Regression
Now we try the same experimet but with different kernel, we use the RBF kernel to perform the parameter tuning on circles dataset.

![circles](circles)

We start with the value of $\alpha = 0.01$ and $\gamma = 0.01$.
> $\gamma$ is defined as $\gamma = \dfrac{1}{2\sigma^2}$. So be careful that a large $\gamma$ corresponds to small $\sigma$ and viceversa.

With this configuration our test error is 13.80%.

![circles_test_error](circles_test_error)

We tune the hyperparameter *alpha* and *gamma* using **GridSearchCV** and after 49 iterations the best params are:
* alpha: 1.0
* gamma: 10

```
#TODO
param_grid = {
    "alpha": [0.001, 0.1, 0.1, 1.0, 10, 100, 1000], # TODO: Insert some values to test
    "gamma": [0.001, 0.1, 0.1, 1.0, 10, 100, 1000] # TODO: Insert some values to test
}
g_model = KernelRidge(kernel="rbf")

g_modelGSCV = model_selection.GridSearchCV(g_model, param_grid)
g_modelGSCV.fit(X_train, y_train)
print(g_modelGSCV.best_params_)
```

To control if the gaussian kernel overfit I used the *validation_curve* and the result is that ***the rbf kernel overfit for values of gamma grather than 10***.

![val_curve_alpha](val_curve_alpha)

![val_curve_gamma](val_curve_gamma)

#### Compare Gaussian and Polynomial kernels on the circles and moons datasets

We use the *polynomial* kernel on the *circle* dataset, the parameter tuning is applied by *GridSearchCV*, the best params are:
* alpha: 1
* degree (of the polynomial kernel): 4
* gamma: 10

The test error is 14%, very similar to test error with *rbf* kernel.

The last task of labs1 is **compare rbf and polynomial kernel on *moons* dataset**.

![moons](moons)

We try different parameter configuration with *RBF* kernel, the first is *alpha* and *gamma* equals to 0.01, the model underfit.

![moons_rbf1](moons_rbf1)

If we increase *gamma* to 0.1 the model perform better, but for *gamma* higher than 1 the model start to overfit.

![moons_rbf2](moons_rbf2)

![moons_rbf_overfit](moons_rbf_overfit)

If we change the kernel from *rbf* to *polynomial* the result is similar, the accuracy score is always near to 99%.

![moons_poly_last](moons_poly_last)

