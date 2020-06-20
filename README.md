### Project Overview

 Lending Club is a peer to peer lending company based in the United States, in which investors provide funds for potential borrowers and investors earn a profit depending on the risk they take (the borrowers credit score).

Lending Club provides the "bridge" between investors and borrowers. This data contains complete loan data for all loans issued for the first quarter of 2012, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Our goal here is to predict potential loan defaulters.

Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. The file is a matrix of about 188183 observations and 77 variables.


### Learnings from the project

 In this project, we'll learn following techniques to apply:

- Multiclass to binary class conversion

- Encoding data

- Filling missing values

- Random forest classifier

- XGBoost classifier



### Approach taken to solve the problem

 To predict the potential loan defaulters we have data from lending club. First, we perform various data cleaning techniques such as feature selection, filling missing values, encoding etc. target feature includes multiple classes so we need to convert them to binary classes and finally, we apply different classification techniques. Here we are applying random forest and XGboost and then check which performs better.


