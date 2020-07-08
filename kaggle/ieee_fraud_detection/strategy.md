# Strategy

## Organizer Notes

We know that 100% of the data is real (non-synthetic) and that it belongs to various types of online payment transactions (CNP). The dataset contains money transfers and other gifting goods and service, like booking a ticket for others, etc.

Certain transactions don't need recipient, so R_emaildomain is null, e.g buying something for yourself self. Due to this, I believe there is merit to treating R_emaildomain.isna() as a _feature_, even though generally, featurizing variable nan counts exhibits crazy inter-train temporal changes as well as train-test covariate shift.

## A step back

Outside of this competition, what would you consider to be some signs of electronic card fraud? These are just notes I am making up, please feel free to add to them and resync to github:

- card used at a location far from user location
- distance from last purcahse location is too far considering the timedelta since last purchase location, when these values are calculated over a user (high cardinality) category.
- transactionamt anomalies, again when grouped by user

## Temporal Issues + CVS

A lot of features in this dataset have drifting distributions between train and test sets. We need to be extra careful of these. In one of my [published notebooks](https://github.com/authman/ieee/blob/master/notebooks/authman/2019-09-01_covariate_shift_and_auc_test_1.ipynb), I calculate the CVS for each variable. We should pick a common sense threshold (like 0.52 for example) and limit ourselves to variables that fall under it. Any variable that has a greater value than this needs to be properly windsorized or otherwise transformed until the feature meets the CVS rating.

I am eager to explore transformations, such as rotations or skewing, as opposed to windsorizations to accomplish this goal.

The transformation needs to be model specific. So when using GBMs, it's important that the feature's trends can be captures in splits along the feature space. The choice of transformation should also take into account if the variable in question is A) continuous vs categorical, and B) Data Type, e.g. date, days, counts, ranks, category, etc.

In addition to computing the CVS, I'd like to look at the actual values of a feature as a function of time after combining both train and test sets. This can be done on a per-value basis for low cardinality variables, or using histograms for high cardinality variables. For example, if there are only 5 values a feature can take:

```
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
traintest['tdt'] = traintest['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
traintest['tmonth'] = traintest.tdt.dt.month
traintest['tyear'] = traintest.tdt.dt.year

# This is just so that the month that overlaps between train + test moves a year
traintest.tyear -= 2017
traintest.tmonth += traintest.tyear*12
traintest = traintest[traintest.tmonth!=18]

tags = []
col = 'my_feature'

look = traintest.groupby(['tmonth', col]).size().reset_index()
cvs = np.round(test_cvs(traintest, col), 4)
plt.title(col + ' CVS: ' + str(cvs))
for colval in traintest[col].unique():
    deeplook = look[look[col]==colval]
    tags.append(plt.plot(deeplook.tmonth, deeplook[0], label=str(colval))[0])

plt.axvline(18, linestyle='--', linewidth=1, c='red') # Gap Month between train/test
plt.legend(handles=tags)
plt.show()
```

Looking at the resulting plot will tell us if the numbre of occurrences of the variable is stable within train, and consistent into test. This is one way of doing this. If the number of values is very large, then we can use the other method the guy in the [heatmap kernel](https://www.kaggle.com/jtrotman/ieee-fraud-time-series-heatmaps/comments) used. This is a very important part of our EDA.  


## Post Processing

So far, we haven't ensembled anything. Much care need be taken when blending predictions to subvert overfitting. Outside of ensembling, another thing we can do to boost our scores is use a good post-processing strategy, driven by posterior knowledge we have on how the dataset was currated.

> Host:
>
> Yes, they're all real data, no synthetic data. The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too. If none of above is reported and found beyond 120 days, then we define as legit transaction (isFraud=0).
> However, in real world fraudulent activity might not be reported, e.g. cardholder was unaware, or forgot to report in time and beyond the claim period, etc. In such cases, supposed fraud might be labeled as legit, but we never could know of them. Thus, we think they're unusual cases and negligible portion.
> It's a complicated situation - usually they will be flagged as fraud. But not all the time afterwards, you can think of one case - the billing address was found to be fraudulent in a past transaction because the credit card associated with it was stolen. But the cardholder is actually the victim, we're not going to blacklist him forever if he uses another legit card for future transaction. There're more other cases but I can't elaborate them all here. One thing we're blacklisting for sure is the card number used for fraud. 

What does this mean? It means that we can create a very simple heuristic or model that fine-tunes our predictions taking into account the labeling logic. So for example, suppose we are able to create a high cardinality feature that identifies a user or a card; if said card is predicted by our level=1 model(s) to be fraud, then any instance of that card after the infrinding timestamp should also have it's isFraud prediction 'upgraded' by some decaying curve.
