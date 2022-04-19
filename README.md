PAC - Polish Abusive Clauses Dataset
============
''I have read and agree to the terms and conditions'' is one of the biggest lies on the Internet. Consumers rarely read the contracts they are required to accept. We conclude agreements over the Internet daily. But do we know the content of these agreements? Do we check potential unfair statements? On the Internet, we probably skip most of the Terms and Conditions. However, it is essential to remember that we conclude many more contracts. Imagine that we want to buy a house, a car, send our kids to the nursery, open a bank account, or many more. In all these situations, you will need to conclude the contract, but there is a high probability that you will not read the entire agreement with proper understanding. European consumer law aims to prevent businesses from using so-called ''unfair contractual terms'' in their unilaterally drafted contracts and require consumers to accept. In this paper the ''unfair contractual term'' is the equivalent of abusive clause, and could be defined as a clause that is unilaterally imposed by one of the contract's parties, unequally affecting the other or creating a situation of imbalance between the duties and rights of the parties.

On the EU and at the national such as the Polish levels, agencies cannot check possible agreements by hand. Hence, we took the first step to evaluate the possibility of accelerating this process. We created a dataset and machine learning models for partially automating the detection of potentially abusive clauses. Consumer protection organizations and agencies can use these resources to make their work more eﬀective and eﬃcient. Moreover, consumers can automatically analyze contracts and understand what they agree upon.

How to run
------------
Using Docker:
```bash
docker build . -t pac-dashboard
docker run -p 8501:8501 pac-dashboard
```
Using Python:
```bash
pip install -r requirements.txt
streamlit run views/dataset_overview.py
```

References
------------

License
------------
MIT licensed
Copyright (C) 2022: [CLARIN-PL](https://github.com/CLARIN-PL)