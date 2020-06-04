
# Hypothesis Testing

# Agenda

1. Establish the basic framework and vocabulary for hypothesis testing
2. Define Null and Alternative Hypothesis
3. Define p-value, alpha, Type 1 and Type 2 Errors
4. Define and perform Z-tests
5. Define and perform T-tests: 1 sample and 2 sample

## Intuition for Hypothesis Testing Example

![heads_or_tails](https://media.giphy.com/media/6jqfXikz9yzhS/giphy.gif)


Jessi has recently claimed that her lucky quarter is actually distinctly different than every other kind of quarter. Due to the unique weight distribution from the quarter's design there is actually a greater chance for the quarter to land tails than other fair coins.

Do we believe her?

I sure don't. But lets be good data scientists and put this claim to the test.

Let's flip the coin once and if it comes up tails then I'll change my mind.

Would you change your mind?

How many tails would I have to flip in order to convince you that this coin actual isn't fair? How many to know for sure that it isn't fair?

What is a reasonable threshold to set?

## Scenarios

- Chemistry - do inputs from two different barley fields produce different
yields?
- Astrophysics - do star systems with near-orbiting gas giants have hotter
stars?
- Economics - demography, surveys, etc.
- Medicine - BMI vs. Hypertension, etc.
- Business - which ad is more effective given engagement?

![img1](./img/img1.png)

![img2](./img/img2.png)

# 1: Establish the basic framework and assumptions for hypothesis testing

### High Level Hypothesis Testing
1. Start with a Scientific Question (yes/no)
2. Take the skeptical stance (Null hypothesis) 
3. State the complement (Alternative)
4. Create a model of the situation Assuming the Null Hypothesis is True!
5. Decide how surprised you would need to be in order to change your mind

## Definitions

**What is statistical hypothesis testing?**

When we perform experiments, we typically do not have access to all the members of a population, and need to take **samples** of measurements to make inferences about the population. 

A statistical hypothesis test is a method for testing a hypothesis about a parameter in a population using data measured in a sample. 

We test a hypothesis by determining the chance of obtaining a sample statistic if the null hypothesis regarding the population parameter is true. 

> The goal of hypothesis testing is to make a decision about the value of a population parameter based on sample data.


**Intuition** 

Suppose you have a large dataset for a population. The data is normally distributed with mean 0 and standard deviation 1.

Along comes a new sample with a sample mean of 2.9.

> The idea behind hypothesis testing is a desire to quantify our belief as to whether our sample of observations came from the same population as the original dataset. 

According to the empirical (68–95–99.7) rule for normal distributions there is only roughly a 0.003 chance that the sample came from the same population, because it is roughly 3 standard deviations above the mean. 

<img src="img/normal_sd_new.png" width="500">
 
To formalize this intuition, we define an threshold value for deciding whether we believe that the sample is from the same underlying population or not. This threshold is $\alpha$, the **significance threshold**.  

This serves as the foundation for hypothesis testing where we will reject or fail to reject the null hypothesis.


# Hypothesis Testing Steps

1. Set up null and alternative hypotheses

2. Choose a significance level (alpha)

3. Determine the critical or p-value (find the rejection region)

4. Calculate the test statistic

5. Compare test-statistic with critical value to accept or reject the Null hypothesis.

### 2: Define Null and Alternative Hypotheses


### The Null Hypothesis

![gmonk](https://vignette.wikia.nocookie.net/villains/images/2/2f/Ogmork.jpg/revision/latest?cb=20120217040244) There is NOTHING, **no** difference.
![the nothing](https://vignette.wikia.nocookie.net/theneverendingstory/images/a/a0/Bern-foster-the-neverending-story-clouds.jpg/revision/latest?cb=20160608083230)

### The Alternative hypothesis

![difference](./img/giphy.gif)

If we're testing the function of a new drug, then the null hypothesis will say that the drug has _no effect_ on patients, or anyway no effect relative to relief of the malady the drug was designed to combat. 

If we're testing whether Peeps cause dementia, then the null hypothesis will say that there is _no correlation_ between Peeps consumption and rate of dementia development.

# One tailed vs two tailed tests

![](img/one_tailed.png)

![](img/two_tailed.png)

## Two Tail Hypothesis

$\Large H_0: \mu_1 - \mu_2 = 0  $  
$\Large H_1: \mu_1 - \mu_2 \neq 0  $
    
## Left Tail Hypothesis

$\Large H_0: \mu_1 < \mu_2  $  
$\Large H_1: \mu_1 >= \mu_2  $
    
## Right Tail Hypothesis
$\Large H_0: \mu_1 > \mu_2  $  
$\Large H_1: \mu_1 <= \mu_2  $

# Write the hypotheses

1. A drug manufacturer claims that a drug increases memory. It designs an experiment where both control and experimental groups are shown a series of images, and records the number of correct recollections until an error of each group. 

Answer  
$

2. An online toystore claims that putting a 5 minute timer on the checkout page of it website decreases conversion rate. It sets up two versions of its site, one with a timer and one with no timer. 

Answer  
$

3. The Kansas City public school system wants to test whether the scores of students who take standardized tests under the supervision of teachers differ from the scores of students who take them in rooms with school administrators.

Answer  
$

4. A pest control company believes that the length of cockroach legs in colonies which have persisted after two or more insecticide treatements are longer than those in which have not been treated to insecticide.

Answer  
$

5. A healthcare company believes patients between the ages of 18 and 25 participate in annual checkups less than all other age groups.

Answer  
$


```python
# Write the hypotheses
'''
A drug manufacturer claims that a drug increases memory. It designs an experiment where both control and experimental groups are shown a series of images, and records the number of correct recollections until an error of each group. 

Null = The mean number of recollections of the control group is greater than or equal to the mean number of guesses of the control group.
Alternative = The mean number of incorrect guesses of the experimental group is greater than the mean number of incorrect guesses of the control group.
'''
'''

An online toystore claims that putting a 5 minute timer on the checkout page of it website decreases conversion rate. It sets up two versions of its site, one with a timer and one with no timer. 

Null = The conversion rate for the timer site is greater than or equal to the conversion rate of the no-timer site
Alternative = The conversion rate for the timer site is less than the conversion rate of the no-timer site
'''
'''

The Kansas City public school system wants to test whether the scores of students who take standardized tests under the supervision of teachers differ from the scores of students who take them in rooms with school administrators.

Null = The scores of students supervised by administrators and students are equal
Alternative = The scores of students supervised by teachers are not equal to the scores of students supervised by administrators.
'''
'''

A pest control company believes that the length of cockroach legs in colonies which have persisted after two or more insecticide treatements are longer than those in which have not been treated to insecticide.

Null = Mean cockroach leg length in both categories of treatment are the same.
Alternative = Mean cockroach leg length in buildings that were treated two or more times are greater than the mean cockroach length of buildings with no treatment.
'''
'''

A healthcare company believes patients between the ages of 18 and 25 participate in annual checkups less than all other age groups.

Null = Patients between the age 18 and 25 and all other age group participate in the same amount of annual checkups.  
Alternative = Patients between the age of 18 and 25 participate in less annual checkups than all other age groups.

'''

```




    '\n\nA healthcare company believes patients between the ages of 18 and 25 participate in annual checkups less than all other age groups.\n\nNull = Patients between the age 18 and 25 and all other age group participate in the same amount of annual checkups.  \nAlternative = Patients between the age of 18 and 25 participate in less annual checkups than all other age groups.\n\n'



# 3. Define p-value, alpha, type I/II errors

## $p$-Values

The basic idea of a p-value is to quantify the probability that the results seen are in fact the result of mere random chance. This is connected with the null hypothesis since, if the null hypothesis is true and there is no significant correlation between the population variables X and Y, then of course any correlation between X and Y observed in our sample would have to be the result of mere random chance.

### How Unlikely Is Too Unlikely?

## $Alpha: \alpha$

Suppose we calculate a p-value for some statistic we've measured (more on this below!) and we get a p-value of 20%. This would mean that there is a 20% chance that the results we observed were the result of mere random chance. Probably this is high enough that we ought _not_ to reject the null hypothesis that our variables are uncorrelated.

In practice, a p-value _threshold_ ($\Large \alpha$) of 5% is very often the default value for these tests of statistical significance. Thus, if it is calculated that the chance that the results we observed were actually the result of randomness is less than 1 in 20, then we would _reject_ the null hypothesis and _accept_ the alternative hypothesis.

If $p \lt \alpha$, we reject the null hypothesis.:

If $p \geq \alpha$, we fail to reject the null hypothesis.

> **We do not accept the alternative hypothesis, we only reject or fail to reject the null hypothesis in favor of the alternative.**

* We do not throw out failed experiments! 
* We say "this methodology, with this data, does not produce significant results" 
    * Maybe we need more data!

**CAUTION** We have to determine our $\alpha$ level, i.e. the threshold for an acceptible p-value, before we conduct our tests. Otherwise, we will be accused of p-hacking.

## Type 1 Errors (False Positives) and Type 2 Errors (False Negatives)
Most tests for the presence of some factor are imperfect. And in fact most tests are imperfect in two ways: They will sometimes fail to predict the presence of that factor when it is after all present, and they will sometimes predict the presence of that factor when in fact it is not. Clearly, the lower these error rates are, the better, but it is not uncommon for these rates to be between 1% and 5%, and sometimes they are even higher than that. (Of course, if they're higher than 50%, then we're better off just flipping a coin to run our test!)

Predicting the presence of some factor (i.e. counter to the null hypothesis) when in fact it is not there (i.e. the null hypothesis is true) is called a "false positive". Failing to predict the presence of some factor (i.e. in accord with the null hypothesis) when in fact it is there (i.e. the null hypothesis is false) is called a "false negative".


How does changing our alpha value change the rate of type 1 and type 2 errors?

# Z-Tests 

![z](https://media.giphy.com/media/4oku9cpYuCNwc/giphy.gif)

A z-test is used when you know the population mean and standard deviation. 
A z-test determines the probability a sample mean,

Our test statistic is the z-stat.

For a single point in relation to a distribution of points:

$z = \dfrac{{x} - \mu}{\sigma}$



for a single data point $x$ is equal to a data point, $\mu$ equals the mean of the standard distribution, and $\sigma$ is the standard deviation of the standard distribution.

<br>Our z score tells us how many standard deviations away from the mean our point is.
<br>We assume that the sample population is normally destributed, and we are familiar with the empirical rule: <br>66:95:99.7

![](img/Empirical_Rule.png)


Because of this, we can say, with a z-score of approximately 2, our data point is 2 standard deviations from the mean, and therefore has a probability of appearing of 1-.95, or .05. 

Example: Assume the mean height for women in the use is normally distributed with a mean of 65 inches and a standard deviation of 4 inches. What is the z-score of a woman who is 75 inches tall? 
    


```python
# your answer here
z_score = (75 - 65)/4
print(z_score)
```

    2.5


When we are working with a sampling distribution, the z score is equal to <br><br>  $\Large z = \dfrac{{\bar{x}} - \mu_{0}}{\dfrac{\sigma}{\sqrt{n}}}$

## Variable review: 

Where $\bar{x}$ equals the sample mean.
<br>$\mu_{0}$ is the mean associated with the null hypothesis.
<br>$\sigma$ is the population standard deviation
<br>$\sqrt{n}$ is the sample size, which reflects that we are dealing with a sample of the population, not the entire population.

The denominator $\frac{\sigma}{\sqrt{n}}$, is the standard error

Standard error is the standard deviation of the sampling mean. We will go into that further, below.

Once we have a z-stat, we can use a [z-table](http://www.z-table.com/) to find the associated p-value.

# Let's first work through a computational method of hypothesis testing.


Let's work with the normal distribution, since it's so useful. Suppose we are told that African elephants have weights distributed normally around a mean of 9000 lbs., with a standard deviation of 900 lbs. Pachyderm Adventures has recently measured the weights of 40 African elephants in Gabon and has calculated their average weight at 8637 lbs. They claim that these statistics on the Gabonese elephants are significant. Let's find out!

What is our null hypothesis?

>


```python
# The mean weight of Gabonese Elephants is equal to the mean weight of African Elephants
```

What is our alternative hypothesis?


```python
# The mean weight of Gabonese Elephants is less than to the mean weight of African Elephants
```


```python
# First, generate the numpy array
np.random.normal(9000, 900, size = 40).mean()
```




    8803.226845423855




```python
random_means = []

for _ in range(1000):
    random_means.append(np.random.normal(9000, 900, size = 40).mean())

random_means[:5]
```




    [8973.860881641354,
     9009.155824283625,
     8970.390116280525,
     9059.896989096258,
     9170.496403788886]



Now let's count how many times the means from the sample distribution were less than the mean weight of the Gabonese elephants.


```python
count = 0
for mean in random_means:
    if mean <= 8637:
        count += 1
        
count/len(random_means)
```




    0.009




```python
def mse(sample, mean):
    sq_errors = []

    for sample_mean in sample:
        sq_errors.append((sample_mean - mean)**2)

    return sum(sq_errors)/len(sample)

rmse = np.sqrt(mse(random_means, 9000))
f"We expect the estimate to be off by {rmse: .2f} lbs on average."

```

Remember we gave the formula for standard error before as $\frac{\sigma}{\sqrt{n}}$
<br> Let's calculate that with our elephant numbers.

Now let's calculate the z-score analytically.
Remember the formula for z-score:
$z = \dfrac{{\bar{x}} - \mu_{0}}{\dfrac{\sigma}{\sqrt{n}}}$

# T-Tests

# z-tests vs t-tests

According to the **Central Limit Theorem**, the sampling distribution of a statistic, like the sample mean, will follow a normal distribution _as long as the sample size is sufficiently large_. 

__What if we don't have large sample sizes?__

When we do not know the population standard deviation or we have a small sample size, the sampling distribution of the sample statistic will follow a t-distribution.  
* Smaller sample sizes have larger variance, and t-distributions account for that by having heavier tails than the normal distribution.
* t-distributions are parameterized by degrees of freedom, fewer degrees of freedom fatter tails. Also converges to a normal distribution as dof >> 0

# One-sample z-tests and one-sample t-tests

One-sample z-tests and one-sample t-tests are hypothesis tests for the population mean $\mu$. 

How do we know whether we need to use a z-test or a t-test? 

<img src="img/z_or_t_test.png" width="500">


**When we perform a hypothesis test for the population mean, we want to know how likely it is to obtain the test statistic for the sample mean given the null hypothesis that the sample mean and population mean are not different.** 

The test statistic for the sample mean summarizes our sample observations. How do test statistics differ for one-sample z-tests and t-tests? 

A t-test is like a modified z-test. 

* Penalize for small sample size: "degrees of freedom"

* Use sample standard deviation $s$ to estimate the population standard deviation $\sigma$.

<img src="img/img5.png" width="500">



### T and Z in detail
![img4](./img/img4.png)

## One-sample z-test

* For large enough sample sizes $n$ with known population standard deviation $\sigma$, the test statistic of the sample mean $\bar x$ is given by the **z-statistic**, 
$$Z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}}$$ where $\mu$ is the population mean.  

* Our hypothesis test tries to answer the question of how likely we are to observe a z-statistic as extreme as our sample's given the null hypothesis that the sample and the population have the same mean, given a significance threshold of $\alpha$. This is a one-sample z-test.  

## One-sample t-test

* For small sample sizes or samples with unknown population standard deviation, the test statistic of the sample mean is given by the **t-statistic**, 
$$ t = \frac{\bar{x} - \mu}{s/\sqrt{n}} $$ Here, $s$ is the sample standard deviation, which is used to estimate the population standard deviation, and $\mu$ is the population mean.  

* Our hypothesis test tries to answer the question of how likely we are to observe a t-statistic as extreme as our sample's given the null hypothesis that the sample and population have the same mean, given a significance threshold of $\alpha$. This is a one-sample t-test.

## Compare and contrast z-tests and t-tests. 
In both cases, it is assumed that the samples are normally distributed. 

A t-test is like a modified z-test:
1. Penalize for small sample size; use "degrees of freedom" 
2. Use the _sample_ standard deviation $s$ to estimate the population standard deviation $\sigma$. 

T-distributions have more probability in the tails. As the sample size increases, this decreases and the t distribution more closely resembles the z, or standard normal, distribution. By sample size n = 1000 they are virtually indistinguishable from each other. 

As the degrees of freedom go up, the t-distribution gets closer to the normal curve.

After calculating our t-stat, we compare it against our t-critical value determined by our preditermined alpha and the degrees of freedom.

Degrees of freedom = n - 1
### T-value table

![img6](./img/img6.png)

We can either look it up (http://www.ttable.org/), or calculate it with python:

Let's go back to our Gabonese elephants, but let's reduce the sample size to 20, and assume we don't know the standard deviation of the population, but know the sample standard deviation to be ~355 lbs.

Here is the new scenario: suppose we are told that African elephants have weights distributed normally around a mean of 9000 lbs. Pachyderm Adventures has recently measured the weights of 20 African elephants in Gabon and has calculated their average weight at 8637 lbs. They claim that these statistics on the Gabonese elephants are significant. Let's find out!

Because the sample size is smaller, we will use a one sample t-test.

Now, let's use the t-table to find our critical t-value.
t-critical = -1.729


So, yes, we can very confidently reject our null.


# Two sample t-test

## Two-sample t-tests 

Sometimes, we are interested in determining whether two population means are equal. In this case, we use two-sample t-tests.

There are two types of two-sample t-tests: **paired** and **independent** (unpaired) tests. 

What's the difference?  

**Paired tests**: How is a sample affected by a certain treatment? The individuals in the sample remain the same and you compare how they change after treatment. 

**Independent tests**: When we compare two different, unrelated samples to each other, we use an independent (or unpaired) two-sample t-test.

The test statistic for an unpaired two-sample t-test is slightly different than the test statistic for the one-sample t-test. 

Assuming equal variances, the test statistic for a two-sample t-test is given by: 

$$ t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{s^2 \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}}$$

where $s^2$ is the pooled sample variance, 

$$ s^2 = \frac{\sum_{i=1}^{n_1} \left(x_i - \bar{x_1}\right)^2 + \sum_{j=1}^{n_2} \left(x_j - \bar{x_2}\right)^2 }{n_1 + n_2 - 2} $$

Here, $n_1$ is the sample size of sample 1 and $n_2$ is the sample size of sample 2. 

An independent two-sample t-test for samples of size $n_1$ and $n_2$ has $(n_1 + n_2 - 2)$ degrees of freedom. 

Now let's say we want to compare our Gabonese elephants to a sample of elephants from Kenya. 

## Example 1
Next, let's finish working through our coffee shop example...  

A coffee shop relocates from Manhattan to Brooklyn and wants to make sure that all lattes are consistent before and after their move. They buy a new machine and hire a new barista. In Manhattan, lattes are made with 4 oz of espresso. A random sample of 25 lattes made in their new store in Brooklyn shows a mean of 4.6 oz and standard deviation of 0.22 oz. Are their lattes different now that they've relocated to Brooklyn? Use a significance level of $\alpha = 0.01$. 

State null and alternative hypothesis

What kind of test? 

Run the test_



```python
'''
State null and alternative hypothesis
1. Null: the amount of espresso in the lattes is the same as before the move.
2. Alternative: the amount of espresso in the lattes is different before and after the move. 

What kind of test? 

* two-tailed one-sample t-test
    * small sample size
    * unknown population standard deviation 
    * two-tailed because we want to know if amounts are same or different 
'''
```


```python
x_bar = 4.6 
mu = 4 
s = 0.22 
n = 25 

df = n-1

t = (x_bar - mu)/(s/n**0.5)
print("The t-statistic for our sample is {}.".format(round(t, 2)))

```

    The t-statistic for our sample is 13.64.



```python
# critical t-statistic values
stats.t.ppf(0.005, df), stats.t.ppf(1-0.005, df)
```




    (-2.796939504772805, 2.796939504772804)



Can we reject the null hypothesis? 




```python
'''
> Yes. t > |t_critical|. we can reject the null hypothesis in favor of the alternative at $\alpha = 0.01$. 
'''
```




    '\n> Yes. t > |t_critical|. we can reject the null hypothesis in favor of the alternative at $\x07lpha = 0.01$. \n'



## Example 2

I'm buying jeans from store A and store B. I know nothing about their inventory other than prices. 

``` python
store1 = [20,30,30,50,75,25,30,30,40,80]
store2 = [60,30,70,90,60,40,70,40]
```

Should I go just to one store for a less expensive pair of jeans? I'm pretty apprehensive about my decision, so $\alpha = 0.1$. It's okay to assume the samples have equal variances.

**State the null and alternative hypotheses**

**What kind of test should we run? Why?** 

**Perform the test.**

**Make decision.**


```python
'''
> Null: Store A and B have the same jean prices. 

> Alternative: Store A and B do not have the same jean prices. 

**What kind of test should we run? Why?** 
> Run a two-tailed two independent sample t-test. Sample sizes are small. 

store1 = [20,30,30,50,75,25,30,30,40,80]
store2 = [60,30,70,90,60,40,70,40]
**Perform the test.**
'''
'''
stats.ttest_ind(store1, store2)
'''

'''
**Make decision.**
> We fail to reject the null hypothesis at a significance level of $\alpha = 0.1$. We do not have evidence to support that jean prices are different in store A and store B. 
'''

```

> We fail to reject the null hypothesis at a significance level of $\alpha = 0.1$. We do not have evidence to support that jean prices are different in store A and store B. 

## Example 3 

Next, let's finish working through the restaurant delivery times problem. 

You measure the delivery times of ten different restaurants in two different neighborhoods. You want to know if restaurants in the different neighborhoods have the same delivery times. It's okay to assume both samples have equal variances. Set your significance threshold to 0.05. 

``` python
delivery_times_A = [28.4, 23.3, 30.4, 28.1, 29.4, 30.6, 27.8, 30.9, 27.0, 32.8]
delivery_times_B = [26.4, 26.3, 27.4, 30.4, 25.1, 28.4, 23.3, 24.7, 31.8, 24.3]
```

State null and alternative hypothesis. What type of test should we perform? 

Run the test and make a decision


```python
'''
> Null hypothesis: The delivery times for restaurants in neighborhood A are equal to delivery times for restaurants in neighborhood B. 

> Alternative hypothesis: Delivery times for restaurants in neighborhood A are not equal to delivery times for restaurants in neighborhood B. 

> Two-sided unpaired two-sample t-test
'''

delivery_times_A = [28.4, 23.3, 30.4, 28.1, 29.4, 30.6, 27.8, 30.9, 27.0, 32.8]
delivery_times_B = [26.4, 26.3, 27.4, 30.4, 25.1, 28.4, 23.3, 24.7, 31.8, 24.3]
stats.ttest_ind(delivery_times_A, delivery_times_B)
'''
> We cannot reject the null hypothesis that restaurant A and B have equal delivery times. p-value > $\alpha$. 
'''
```




    '\n> We cannot reject the null hypothesis that restaurant A and B have equal delivery times. p-value > $\x07lpha$. \n'



# Level Up: More practice problems!

A rental car company claims the mean time to rent a car on their website is 60 seconds with a standard deviation of 30 seconds. A random sample of 36 customers attempted to rent a car on the website. The mean time to rent was 75 seconds. Is this enough evidence to contradict the company's claim at a significance level of $\alpha = 0.05$? 

Null hypothesis:

Alternative hypothesis:


Reject?:

Consider the gain in weight (in grams) of 19 female rats between 28 and 84 days after birth. 

Twelve rats were fed on a high protein diet and seven rats were fed on a low protein diet.

``` python
high_protein = [134, 146, 104, 119, 124, 161, 107, 83, 113, 129, 97, 123]
low_protein = [70, 118, 101, 85, 107, 132, 94]
```

Is there any difference in the weight gain of rats fed on high protein diet vs low protein diet? It's OK to assume equal sample variances. 

Null and alternative hypotheses? 

> null: 

> alternative: 

What kind of test should we perform and why? 

> Test:

We fail to reject the null hypothesis at a significance level of $\alpha = 0.05$. 

**What if we wanted to test if the rats who ate a high protein diet gained more weight than those who ate a low-protein diet?**

Null:

alternative:

Kind of test? 

Critical test statistic value? 

Can we reject?

# Summary 

Key Takeaways:

* A statistical hypothesis test is a method for testing a hypothesis about a parameter in a population using data measured in a sample. 
* Hypothesis tests consist of a null hypothesis and an alternative hypothesis.
* We test a hypothesis by determining the chance of obtaining a sample statistic if the null hypothesis regarding the population parameter is true. 
* One-sample z-tests and one-sample t-tests are hypothesis tests for the population mean $\mu$. 
* We use a one-sample z-test for the population mean when the population standard deviation is known and the sample size is sufficiently large. We use a one-sample t-test for the population mean when the population standard deviation is unknown or when the sample size is small. 
* Two-sample t-tests are hypothesis tests for differences in two population means. 
