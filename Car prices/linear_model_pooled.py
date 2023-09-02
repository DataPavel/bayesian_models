import pandas as pd
import numpy as np
from scipy import stats

import pymc as pm
import arviz as az


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class CarPrices():

    def __init__(self, seed=None):

        self.seed=seed

    def fit(self, df, test_size=0):

        scaler = MinMaxScaler()
        if test_size==0:
            scaler.fit(df)
            self.df_train_scaled = scaler.transform(df)
        else:

            self.df_train, self.df_test = train_test_split(df, test_size=test_size, random_state=self.seed)
            scaler.fit(self.df_train)
            self.df_train_scaled = scaler.transform(self.df_train)
        self.scaler = scaler

        with pm.Model() as demio_multiple:

            # data
            demio_price = pm.MutableData('demio_price', self.df_train_scaled[:,-1])
            X = pm.MutableData('covariates', self.df_train_scaled[:,:-1])

            # prior
            # intercept = pm.SkewNormal('intercept', mu=0.5, sigma=0.25, alpha=-2)
            intercept = pm.Beta('intercept', alpha=1.5, beta=6) # good for gen_1 cars
            # intercept = pm.Normal('intercept', mu=0.5, sigma=0.25)
            β = pm.Normal('β', mu=0, sigma=0.1, shape=self.df_train_scaled[:,:-1].shape[1])

            # linear regression
            μ = pm.Deterministic('μ', (intercept
                                       + β[0] * X[:,0]
                                       + β[1] * X[:,1]
                                       + β[2] * X[:,2]
                                       + β[3] * X[:,3]
                                       + β[4] * X[:,4]))

            # observational noise
            ε = pm.HalfNormal('ε', 0.05)

            # likelihood
            price = pm.Normal('price', mu=μ, sigma=ε, observed=demio_price)


        with demio_multiple:

            trace_demio_oos = pm.sample(
                # draws=1000,
                # tune=2000,
                target_accept=0.95
            )

            trace_demio_oos.extend(pm.sample_prior_predictive())
            trace_demio_oos.extend(pm.sample_posterior_predictive(trace_demio_oos))

        self.demio_multiple = demio_multiple
        self.trace_demio_oos = trace_demio_oos


        return self

    def check_on_test(self, df):

        df_test_scaled = self.scaler.transform(df)

        with self.demio_multiple:

            pm.set_data(
                {
                    'covariates': df_test_scaled[:,:-1]
                }
            )
        with self.demio_multiple:
            m = pm.sample_posterior_predictive(
                self.trace_demio_oos,
                predictions=True,
            )
        ax = az.plot_posterior(
            (m.predictions['price'] * self.scaler.data_range_[-1] + self.scaler.data_min_[-1]) ** 2,
            ref_val=(df['price_eur_sqrt'] ** 2).to_list(),
            # var_names=
            hdi_prob=0.94,
            # ax=ax[0],
            textsize=12,
            )
        return ax




    def make_predictions(self, df, plot_title='fill in your plot name'):


        preds_scaled = self.scaler.transform(df)

        with self.demio_multiple:

            pm.set_data(
                {
                    'covariates': preds_scaled[:,:-1]
                }
            )
        with self.demio_multiple:
            m = pm.sample_posterior_predictive(
                self.trace_demio_oos,
                predictions=True,
                            )
        ax = az.plot_posterior(
            (m.predictions['price'] * self.scaler.data_range_[-1] + self.scaler.data_min_[-1]) ** 2,
            hdi_prob=0.94,
            textsize=12,
        );
        ax.set_title(plot_title)


        return ax

    # Add representation of the class
    def __repr__(self):
        return f"CarPrices(seed={self.seed})"
