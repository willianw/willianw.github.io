---
title:  "Political compass test"
subtitle: "(or a tutorial for your first Data Science project)"
category: "Data Science"
image: "cover.png"
tags: ["politics"]
---

One of the most fascinating things regarding to Data Science is the huge capacity it provides us to answer questions — maybe “answer” is very pretentious: let’s say “to provide insights”.

# Introduction

Data Science applies several methods coming from sciences, with some fundamental differences:

- **Data:** traditional experiments are usually very expensive in the data collection part: involve laboratories, cell culture, reactors which generate high energy amounts, sometimes even animal sacrifice, among others. Currently, with the internet, we can reuse data and combine fonts, allowing much more knowledge production;
- **Models:** today with the great processing power of computers, models are able to detect patterns without us having to provide hypothesis. This facilitates the process of conduction of an experiment because we can select in a much faster way which patterns deserve a more detailed analysis and which we may discard.

That said, I propose here an experiment to be described in a very detailed way which refers to **politics**. There are several reasons that make this matter very interesting for a Data Science project:

- This subject is usually treated in a very subjective manner - always influenced by third parts opinions, analysis over analysis, and a lot of sensationalism, making the information to come always in a very distorted way. Maybe it is interesting to listen what data has to say;
- As it is a matter of general interest - some more than others, but everyone somehow uses up this kind of information - we have a wide data base, which makes the analysis very interesting;
- Similarly, because it's a matter of general interest, we have a certain knowledge base, albeit common sense, allowing us to check if data is consistent or not - we know, though roughly, how some groups behave, who is closer or further away from whom and which topics are more or less important to certain political sides;
- And eventually a very practical aspect: the data that I found are well structured. This means that they can be extracted in a very systematic manner, i.e., dataset is easy to get.

There are several things we can identify, such as: "which are the politically fiercest periods of the year", "which parties vote accordingly, and which ones have a bigger divergence among its congressmen", or "which is the party that vote most differently from the others". All these questions are subject of analysis, and is my intent to run by some of them.

***

Time for hands on! The whole project and the steps shown here are documented in the GitHub page. To understand the process, it is best that you have some Python or general programming knowledge. For simplicity, demonstrations done here are from Senate, as the Chamber of Deputies is very similar, avoiding thus redundancies. But one may check the Chamber of Deputies too in the GitHub project.

Let’s start by the data source.

# Part 1 — Obtaining data

The data were extracted from the section of Chamber of Deputies and Senate about open data. In them there are several items — as budget expenditure, hirings, and even some tutorials about hot to access the data.
The way of obtaining data was through Web Scraping, accessing these portals and saving the obtained content. As a matter of example, we’re going to access the page of Senate schedule, on 01/01/2019. This page shows the content of the month from the given day. In this case, it shows the activities of January-2019, from the 1st day.

![Agenda](agenda.png)

The first spotlight in yellow is the event description because this the only needed parameter. We can use the same url changing just this date in order to obtain the whole Senate schedule in several years.
The second spotlight in yellow is the date, that is the reception of the President of the Republic. In this case, for instance, there was no vote — as a matter of fact, the parliamentary activities begin in February. Let’s see a case that includes voting.

![Session](session.png)

Here we have in schedule a proposed constitutional amendment (PEC 25/2017), which was voted in February. There are many descriptions about the item, as the proposal date, the appraisal (discussion session or voting), among others. But here there is not a list of senators and their votes. In the schedule there is only the project descriptions, but these projects usually involve discussions and votings, that occur in different dates. The voting description is in another page, but we need to pass in this page to obtain the code, highlighted in orange. From it we search the votes specifically.

![Matter](matter.png)

Here we have the page of this matter. There is initially an abstract about the voting, showing its result (unanimous victory!). After that we have the parliamentary votes, with all data about the senator and at the end his vote. We see, for example, that the senator Jarbas Vasconcelos voted favourably to this PEC.
In order to effectively process these data, I’ve used a very famous tool for this kind of activity: Scrapy. It has a set of tools that make very simple accessing pages, extracting relevant content and saving results.

![Start Requests](start_requests.png)

We have initially the url of the Senate schedule. The variables are the month and the year, which starts at 04/2011. This is because from this date data is available in the API. Then for each month of each year, we make a request, which will be processed by the function parse_agenda.

![Parse Agenda](parse_agenda.png)

This function is wide and made exclusively to access the Senate schedule. But this is common. Each site has its particularities, as the manners of showing information, the ways of navigation between links, etc. There are even sites that protect themselves against who tries to access their content through these robots, using using tools as the reCAPTCHA. However, as in our case we use an open and public database, there is no need to worry about this.

# Part 2 — Exploratory Analysis

Here start charts and visualisations. Let’s start with a very simple one:

![Votes from each party](votes_from_each_party.png)

The chart above tells us very little. This is a history since 2011 and parties change quite often — switch initials, separate, attach themselves, etc. More interestingly would be to see this over time.

![Votes Over Time](votes_over_time.png)

Even so it seems not informative. There are many small parties, with few senators, that spoils visualisation. In order to avoid this, let’s make a cut, keeping the parties with more votes and renaming others to ‘others’. Another change that improves visualisation is proportionality, i.e., instead of watching the total of votes from a party in a set year, we’ll see the percentage of participation in the period. Retaining only the parties with more than 1000 votes altogether, we stay like:

![Votes Over Time Normalized](votes_over_time_normalized.png)

# Part 3 — Principal Component Analysis

Things here start to become interesting.

PCA is a widely used method emerged from “tradicional sciences” for, among other things, visualisation and interpretation of results. For whom already understands it a little, it’s the decomposition of the covariance matrix in its eigenvectors, sorted by its eigenvalues, which corresponds to the variance of data in each eigenvector.

![PCA Example](pca_example.png)

For whom doesn’t understands it yet, see the figure beside. We have a distribution in two dimensions. Note however that that base formed by those little arrows is far more interesting for explaining data than the x and y axes. More interesting than that: as one arrow is considerably bigger than the other, we can say that it is the **principal component**. Thus a good approximation we can do is to consider just the bigger arrow to represent data, simplifying analysis. And is that what we’ll do with our parties:

![Matrix Correlation](correlation_matrix.png)

This matrix corresponds to the correlation between parties votes. The greener the color between two parties, the more they vote accordingly. See that the point of a party with itself is always the greenest possible (evidently). In yellow are parties with null correlation — i.e., given the vote of one of them, it is not possible to know the other one’s vote — and in red are the ones with opposite correlation. Noteworthy is that there is not a strongly negative correlation between two parties (there are some points which are slightly red, but they do not even compare to the strength of the green ones). This means that political polarisations are not exactly due to the vote for or against determined matters, but rather by the prioritisation of these.
Here I show you another version of that matrix, with a few modifications:

![Modified Matrix Correlation](modified_correlation_matrix.png)

There were two changes: the cut of parties (keeping only the ones which have more than 1000 votes) and the change of color scale: in the first chart, the red-yellow-green scale lies on [-1.0, 0.0, 1.0]; in the second one, it lies on [0.0, 0.5, 1.0]. And here we have already learned an important lesson:

>The way data are shown changes drastically their interpretation

See that with this second char, it’s much more difficult to tell there is no political polarisation. One can see very clearly ‘friend parties’ and ‘enemy parties’. Shortly, is very easy to use data to tell your own version of facts. Be careful!
Something else that is clearly visible on both graphs is the arising of political clusters, some blocks that use to vote in a cohesive way.
See, for example, the groups PSDB and DEM; PP, PL and MDB; and PT and PCdoB. Data shows us something very simple: the coherence between these parties. Coupling this to our specific knowledge, we have a more interesting knowledge production. I’m not a politics specialist, but I dare to affirm that this is a manifestation of what we call respectively of right-wing, centre and left-wing. Let’s deepen this analysis.

# Part 4 — The political compass

![Meme](meme.jpeg)

One of the things I set out to do is to know if it is possible — without manipulating data — to visualise a very popular map, expressed by the meme above.

The PCA idea comes willingly to help us in this task. It will hardly appear a so simple thing, but I dare try to reach some level of similarity. The process is very simple: just do the same analysis, but instead of obtaining the principal component, one obtains the two principal components.
No further delay, let’s go to results:

![PCA1](pca1.png)

![PCA2](pca2.png)

The x coordinate is the same used in the previous part, with the difference that I’ve multiplied it by -1 (pure aesthetics, so that left-wing parties stay left and right-wing parties stay right).
As to the y coordinate, it seems that if we assume it’s about the authoritarian/liberal axis, it’ll look like this picture:

![BBC](bbc.jpeg)

Of course there are several caveats to be done comparing my map with the BBC’s:

- They use a well defined law set to affirm what is liberal, authoritarian, left-wing and right-wing; while I use just PCA;
- BBC uses deputies data while I use senators data;
- These laws are from years between 2015–2017, while the range of Senate data is 2011–2019.

However, it’s very exciting to achieve good results with a relatively simple analysis. This is also a great lesson:

>Sometimes it’s possible to achieve good results using just traditional statistical techniques.

When it comes to data analysis, the more tools we dispose, the better. Let’s apply a little Machine Learning!

# Part 5 — Parties classification

The following table shows the relation between senators, his parties and voting.

![Votes by Senate Table](votes_by_senator_table.png)

The numbers in rows are the senators identifiers, according to Senate official API. The first column is the party to which each senator belongs and the next columns are the matters identifiers — law proposals, changes in specific laws, etc. This is about a very simple classification problem: we have the columns that represent votings, and the target column, with the senator’s party. Here I’ve used XGBoost, an implementation of the famous Gradient Boosting algorithm, because it’s fast and it gives some interesting visualisations. The code snippet is very simple. I’ve used cross validation in order to have a good idea of accuracy.

![XGBoost](xgboost.png)

The results, however, are terrible:
```0.15625, 0.16129032, 0.24590164, 0.37288136, 0.32142857```

In a Data Science project, things go wrong. More important that making them work, is understanding why they have failed. In this case the answer is very simple: (302, 832). Those are the data frame dimensions. There are 302 rows and 832 columns. No need to hesitate too much to state that it is a clear case of overfitting. To confirm this, just look to this confusion matrix:

![Confusion Matrix](confusion_matrix.png)

This matrix was made with the whole dataset. This means the model was very capable of learning data it have seen, but not too much for data it hasn’t seen yet. There are a lot of techniques to improve this model — as reducing the number of columns, use regularisation, adjust hyper-parameters or just switch to a simpler model. But in the moment this doesn’t matter. There are interesting things model has to show us.

![Most Important Projects](most_important_projects.png)

Next we have the five most important projects to define the political positioning of a senator, and are they:

- **PLV 15/2013** — Reduces to zero the aliquots of the Contribution to PIS / PASEP, COFINS, Contribution to PIS / PASEP — Import and COFINS — Importation on the revenue from sales in the domestic market and on imports of products that make up the Basic-needs grocery package;
- **SCD 5/2015** — Regulates art. 7th, single paragraph, of the Federal Constitution (Constitutional Amendment nº 72, 2013, origin: PEC №66, of 2012 — PEC of the housekeepers);
- **MPV 695/2015** — Authorizes Banco do Brasil SA and Caixa Econômica Federal to acquire participation in the terms and conditions set forth in art. 2 of Law nº 11,908, of March 3, 2009, and makes other provisions;
- **MPV 696/2015** — Extinguishes and transforms public offices and changes Law nº 10,683, of May 28, 2003, which provides for the organization of the Presidency of the Republic and of the Ministries;
- **PLV 7/2012** — Reduces to zero the rates of the Contribution to the PIS / Pasep, the Contribution for the Financing of Social Security — COFINS, the Contribution to the PIS / Pasep — Import and the Cofins — Imports incident on import and sales revenue in the domestic market of the products it mentions;

# Conclusion

Data Science is about, in the end, exploring possibilities, testing hypothesis, proposing solutions. There aren’t many fixed pattern to be followed — creativity matters a lot. I deeply hope that this article have helped you to think about new ideas and explore new limits. See the code!

{% include repo.html name='brazilian-political-compass' %}

# Credits and References
- [Post at Medium](https://medium.com/neuronio/political-compass-test-or-a-tutorial-for-your-first-data-science-project-ff7fbe5112c5)
- Photo by Jørgen Håland on Unsplash
