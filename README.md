# Wolfer-Number

---
title: "ECO 4051 Financial Econometrics"
subtitle: "Final Project: Wolfer Number"
author: "Rahul Bakshi, Alex Dimcevski, and Lev Gofman"
output: word_document
---

```{r Setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, 
                      eval = TRUE,
                      warning = FALSE,
                      message = FALSE,
                      comment = "",
                      out.width = "70%",
                      fig.asp  = 0.5,
                      fig.align = "center")

setwd("/Users/Lev/Downloads")
```

## Introduction:

For our research, we have chosen to study the Wolf/Wolfer Number. This number states the number of sunspots that are present on the surface of the sun. This number is primarily important because the solar activity affects things such as satellite drag, telecommunications outages, hazards in connection with occurrence of strong solar wind streams producing blackouts of power plants, and prediction of radiation risk as high powerful radiation can lead to computer and computer memory upsets or failures on manned space flights.

There hasn't been a model that has been collectively agreed upon as being the best for determining the forecasted amount of sunspots in the upcoming years. There have been a wide range of proposed models ranging from finding various prediction methods and comparing the root mean square method, to AR models, to even modelling with computers by defining neural networks which generates models from simple to complex ones until the testing accuracy increases.

## Survey of the literature:

The first article that we will discuss is the "Sunspot Number Prediction by an Autoregressive Model", written by R. Werner. For his data, he used the sunspot numbers available from 1749 up to 2010 for his model. He used the data to construct an ARMA model that is based on the Box-Jenkins method. He used Statistica 6 program to determine the model parameters of the AR model which returned that the best model to use was AR(9).

The next article that will be discussing is "The Solar Cycle", by David H. Hathaway. Instead of looking at the sunspot numbers directly, the main focus was on the cycles of the sunspots. The cycles were characterized by features such as their maxima and minima, periods and amplitudes, and cycle shape. A variety of prediction techniques are then applied for the cycles, and finding which has the least mean square error. Conclusion of this was that a solar cycle has a period of about 11 years with standard deviation of about 14 months, solar cycles are assymetric by their rise to maximum is shorter than decline to minimum. 

In "Sunspot numbers: data analysis, predictions and economic impacts", by A. Gkana and L. Zachilas, to forecast the peak of the next cycle, they used the software GMDH Shell, which is a predictive modeling tool that produces mathematical models and makes predictions. With the polynomial neural networks, models are generated ranging from simple to complex models and selects based on the lowest root mean square error. The model that they've gotten for predicting the value of the monthly sunspot at a specific time period t isn't a simple model. The data that they analyzed was the monthly sunspot number from January 1749 to June 2013 and the exact model is shown below, with the timestep being months. The model estimates relying on the number of sunspots from including 2098 months ago (almost 175 years before that date).

![Number of sunspots at time = t](capture.PNG)

## Model:

The model that we will be using is an AR model. We are using this model as based on the plot of the average number of sunspots in a given year, the data is cyclical. The cyclical data means we may be able to use an autoregressive model to predict the number of sunspots as it may be based on past behavior. 

## Data:

The source of the data is http://web.mit.edu/r/current/lib/R/library/datasets/html/sunspot.year.html. The variables that we received in this data set is year and the average number of sunspots in that year. The period that the data goes from is 1700 to 1998.

## Empirical Application:


```{r Packages}
library(tidyverse)
library(TTR)
library(rugarch)
library(quantmod)
library(dyn)
library(gridExtra)
library(FitAR)
```

```{r Functions}
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.03, .97), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

ggacf <- function(y, lag = 12, plot.zero="no", alpha = 0.05)
{
T <- length(y)
y.acf <- acf(y, lag.max=lag, plot=FALSE)
if (plot.zero == "yes") y.acf <- data.frame(lag= 0:lag, acf = y.acf$acf)
if (plot.zero == "no") y.acf <- data.frame(lag= 1:lag, acf = y.acf$acf[-1])
library(ggplot2)
ggplot(y.acf, aes(lag, acf)) + geom_bar(stat="identity", fill="orange") +
geom_hline(yintercept = qnorm(1-alpha/2) * T^(-0.5), color="steelblue3", linetype="dashed") +
geom_hline(yintercept = qnorm(alpha/2) * T^(-0.5), color="steelblue3", linetype="dashed") +
geom_hline(yintercept = 0, color="steelblue3") +theme_classic() + ylab("") + ggtitle("ACF")
}
```

Here we load the data and put the data into a time series:

```{r Data}

data <- read_csv("data.csv") %>% na.omit()
names(data) <- c("year", "times")

data_df = data.frame(date=data$year, coredata(data$times))
names(data_df) <- c("year", "times")
data_df$year <- as.character(data_df$year)
data_df$year <- as.Date(data_df$year, format = "%Y")

data_xts <- xts(data_df[,-1], order.by = data_df[,1])
```
Plot of the graph: 

```{r Graph}

x_tick <- seq(1700, 1988, 20)

ggplot(data, aes(year, times, group= 1)) + geom_line(col= "blue") + scale_x_discrete(breaks=x_tick,
        labels=x_tick) + theme_minimal()
```

From the graph above, we can see that there is a cyclical pattern. Also the data appears to be stationary as there isn't an upward or downward trend and there is no seasonality. 

```{r lm}

fit <- lm(data= data, year ~ times)

summary(fit)
plot(fit)
abline(fit)
```

Even though the linear model shows that it is statistically significant, a linear model isn't appropriate here as if you look at the residuals versus fitted plot, there are too many high residuals in the plot. So we will move towards looking at an AR model.


```{r forecast}

ar_model <- SelectModel(data$times, lag.max=24, Criterion = "BIC", Best=1)

fit_model <- arima(data$times, order = c(ar_model, 0, 0))
fit_model
ciao <- forecast::forecast(fit_model, h = 50 )
# print a table with the forecasts
knitr::kable(as.data.frame(ciao), digits=3)

autoplot(ciao)
plot(ciao)
realdata <- read_csv("realdata.csv")
lines(realdata,col="red")
```

Finding the AR model using the BIC as the Criterion, we have an AR model of AR(9). We can use this model for forecasting, which we have done for the next 50 years. As the data we used is from 1700 to 1998, we can compare to more current data. We downloaded the average number of sunspots in a year till 2014 from http://www.sws.bom.gov.au/Educational/2/3/6, and plotted it in red along the forecasted results. The forecasted results followed close to the actual numbers. 

```{r GGACF}
ggacf(residuals(fit_model))
```

Looking at the autocorrelations of rhe residuals however, there is a pattern so to to try to remove the pattern, we will try to take the square root of the data and see how the result will look like.

```{r Graph2}

data$times <- sqrt(data$times)
x_tick <- seq(1700, 1988, 20)

ggplot(data, aes(year, times, group= 1)) + geom_line(col= "blue") + scale_x_discrete(breaks=x_tick,
        labels=x_tick) + theme_minimal()
```


```{r forecast2}

ar_model <- SelectModel(data$times, lag.max=24, Criterion = "BIC", Best=1)

fit_model <- arima(data$times, order = c(ar_model, 0, 0))
fit_model
ciao <- forecast::forecast(fit_model, h = 50 )
# print a table with the forecasts
knitr::kable(as.data.frame(ciao), digits=3)

autoplot(ciao)
plot(ciao)
sqrtrealdata <- read_csv("sqrtrealdata.csv")
lines(sqrtrealdata,col="red")
```

From the square root of the forecasted and the square root of the actual data, the forecast that we have still follows along with the real data. 

```{r GGACF2}
ggacf(residuals(fit_model))
```

From the autocorrelation of these residuals, there still is a bit of a pattern from residuals 6 to 10; however, it is a bit better than before. 

## Conclusion:

As the forecast can approximate when there will be the fewest sunspots during a cycle, we can predict when manned space flights would be best to launch to have a lower chance of computer failures. Similar to the article by R. Werner, we also received that the best model to use was AR(9), even though the data we used was at a different time period.

## Works Cited:

Gkana, A.and Zachilas, L. *"Sunspot numbers: data analysis, predictions and economic impacts"*

Hathaway, David H. *"The Solar Cycle"*
