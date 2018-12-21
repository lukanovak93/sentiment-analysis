# sentiment-analysis

This is a simple project that takes on sentiment analysis. I'm going to try a lot of different methods, embeddings and approaches and see what gives the best results.

### Preprocessing

Sample text before preprocessing:  
`Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!`

  #### 1. Simple
   - lowercase everything
   - remove punctuation
   - remove multiple whitespaces  

      The same text sample after applying this preprocessing:  
    `bromwell high is a cartoon comedy it ran at the same time as some other programs about school life such as teachers my 35 years in the teaching profession lead me to believe that bromwell highs satire is much closer to reality than is teachers the scramble to survive financially the insightful students who can see right through their pathetic teachers pomp the pettiness of the whole situation all remind me of the schools i knew and their students when i saw the episode in which a student repeatedly tried to burn down the school i immediately recalled  at  high a classic line inspector im here to sack one of your teachers student welcome to bromwell high i expect that many adults of my age think that bromwell high is far fetched what a pity that it isnt`


  #### 2. Standard
   - lowercase everything
   - remove punctuation
   - remove multiple whitespaces    
   - remove stopwords


  #### 3. Advanced
   - lowercase everything
   - remove punctuation
   - remove multiple whitespaces    
   - remove stopwords
   - lemmatization/stemming


### Benchmarks

|  Preprocessing  |  Embedding  |         Model         |  Accuracy  |
|:---------------:|:-----------:|:---------------------:|:----------:|
|     Standard    |   One-hot   |  Logistic Regression  |   0.88152  |
|     Standard    |   One-hot   |          SVM          |   0.87112  |

### Setup

I'm using Anaconda Python 3.7.1. All the other packages are in `requirements.txt`

To install the required packages just run `pip3 install -r requirements.txt` and all of the packages should be installed for you.

### Running the training

If you want to run the training of a model, here is the usage (which you can get by typing `pytohn3 main.py -h`):

    usage: main.py [-h] -m {logistic,svm} [-v]

    required arguments:
    -m {logistic,svm}, --model {logistic,svm}
                          Specify which model to use
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Verbose output

### TODO
 - [ ] Stemming/Lemmatizing
 - [ ] n-grams
 - [ ] TF-IDF
 - [ ] Word2Vec/GloVe
 - [x] SVM
