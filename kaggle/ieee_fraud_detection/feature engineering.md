# Feature Engineering

I've already put out a few markdown files / notebooks on the strategy we should use and validation methods. I wanted to conclude with some notes on feature engineering. There will also be a notebook accompanying this. I belive this competition will be won by the teams with the best ED rather than AutoML type solutions. Or maybe, the teams with the best EDA _combined_ with AutoML to take it to the next level. To that end, there are many gold placed teams that have very few (20-40) features. This means:

1. They aren't overfit since they used so few features, there's a good chance their swolutions will generalize well
2. There's room for a lot of power in a few good features rather than building out 9023923902390 garbage ones!

How does one go about building good feature? By understanding what the features are / through EDA. Let's take a moment to talk about some of the features. For example, the M-variables. We know these are matches:

- Did the name on the card match the name provided?
- Did the address on the card match the provided address?
- etc?

What features and interactions make sense here? I think calculating counts makes a lot of sense. For example, when M1 = T, calculate the counts of transactions in the last 7 day window grouped by a categorical variable:

- Count of M1=T transactions in the past week
- Count of M1=T transactions having the sample in question's addr in the past week
- Count of M1=T transactions having the sample in question's card1-6 in the past week
- Count of M1=T transactions having the sample in question's productcd in the past week
- Count of M1=T transactions having the sample in question's r/e email domain in the past week
- Same thing with other categorical variable, such as Operating System, or clustering...
- etc.

If we find that the count of transactions for a 1-week period produces too small results for any of the categorical variables we examine, we should increase the window to capture more signal.

How about another feature like transactionamt, what can do with that?

- For transaction amount, it makes more sense to calculate mean, median, sample diff to mean, sample diff to median, etc type feature interactions.
- addr in last week
- card cols in last week
- productcd in last week
- Pemaildomain Remaildomain in last week
- other combined categoricals, e.g. card1+card2+card3+... in longer time spans (month?)
- other broken categoricals, e.g. 'Android' built from clustering other columns in shorter time spans (days?)

So in the above, for example, we could try calculating mean transactionamt for a specific addr in the past 7 days. Check how many values come out of that. If there are so few (1:1) that the answer to this is just the transactionamt itself, then clearly we need to increase the window. More than just calcualting the mean transactionamt within the window, we can subtract the sample in question's transactionamt from this mean to see how far the sample is from the mean.

## Creating Categorical Variables (by Data Type)
- Create innergroups within categoricals that have many unique value counts
    - e.g. card1 has 18k values. Derive how to group different card1 values, e.g. close count_encoded values.

- Break apart categoricals that have too few unique value coiunts
    - e.g. ProductCD only has 5 potential values. It'd be nice if we could find a way to break apart the 'C' category into multiple children. Maybe some type of hierarchical clustering / similarity classification within unique value, etc; whatever technique we use should ignore the nans...

- Create categorical groups that span different categorical variables, e.g. card1 + card2 + card3 in string format.