# MOVIE RECOMMENDER SYSTEM

 we'll develop a very simple movie recommender system in Python that uses the correlation between the ratings assigned to different movies, in order to find the similarity between the movies.The dataset that we are going to use for this problem is the MovieLens Dataset . https://grouplens.org/datasets/movielens/latest/  .It contains a subset of the actual movie dataset and contains 100000 ratings for 9000 movies by 700 users.


```python
import numpy as np
import pandas as pd
```


```python
ratings_data=pd.read_csv("ratings.csv")
```


```python
ratings_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_names=pd.read_csv("movies.csv")
```


```python
movie_names.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, this dataset contains movieId, the title of the movie, and its genre. We need a dataset that contains the userId, movie title, and its ratings. We have this information in two different dataframe objects: "ratings_data" and "movie_names". To get our desired information in a single dataframe, we can merge the two dataframes objects on the movieId column since it is common between the two dataframes.


```python
movie_data=pd.merge(ratings_data,movie_names,on='movieId')
```


```python
movie_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>4.0</td>
      <td>847434962</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1</td>
      <td>4.5</td>
      <td>1106635946</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>2.5</td>
      <td>1510577970</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>1</td>
      <td>4.5</td>
      <td>1305696483</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>
</div>



## Now let's take a look at the average rating of each movie.


```python
movie_data.groupby('title')['rating'].mean().head()
```




    title
    '71 (2014)                                 4.0
    'Hellboy': The Seeds of Creation (2004)    4.0
    'Round Midnight (1986)                     3.5
    'Salem's Lot (2004)                        5.0
    'Til There Was You (1997)                  4.0
    Name: rating, dtype: float64



## Ratings in the descending order of their average ratings:


```python
movie_data.groupby('title')['rating'].mean().sort_values(ascending=True).tail()
```




    title
    Winnie the Pooh and the Day of Concern (1972)    5.0
    Sorority House Massacre II (1990)                5.0
    My Love (2006)                                   5.0
    Winter in Prostokvashino (1984)                  5.0
    Karlson Returns (1970)                           5.0
    Name: rating, dtype: float64



The above table shows the list of movies with the average rating of 5 amongst all the movies in the given dataset.The movies have now been sorted according to the ascending order of their ratings. However, there is a problem. A movie can make it to the top of the above list even if only a single user has given it five stars. Therefore, the above stats can be misleading. Normally, a movie which is really a good one gets a higher rating by a large number of users.


```python
movie_data.groupby('title')['rating'].count().sort_values(ascending=True).tail()
```




    title
    Matrix, The (1999)                  278
    Silence of the Lambs, The (1991)    279
    Pulp Fiction (1994)                 307
    Shawshank Redemption, The (1994)    317
    Forrest Gump (1994)                 329
    Name: rating, dtype: int64



Now you can see some really good movies at the top. The above list supports our point that good movies normally receive higher ratings. Now we know that both the average rating per movie and the number of ratings per movie are important attributes. Let's create a new dataframe that contains both of these attributes.

## Creating a new dataframe ratings_mean_count to store title,average ratings and counts on a movie


```python
ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
```


```python
ratings_mean_count['rating_counts'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())
```


```python
ratings_mean_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>4.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
%matplotlib inline

plt.figure(figsize=(8,6))
plt.xlabel('Number of ratings')
plt.ylabel('Number of movies')
plt.title('RATINGS COUNT VS NUMBER OF MOVIES')
ratings_mean_count['rating_counts'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.xlabel('Average ratings')
plt.ylabel('Number of movies')
plt.title('AVERAGE RATINGS VS NUMBER OF MOVIES')
ratings_mean_count['rating'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1df12ad5408>




![png](output_21_1.png)



![png](output_21_2.png)


# Finding Similarities Between Movies


```python

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>4.0</td>
      <td>847434962</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1</td>
      <td>4.5</td>
      <td>1106635946</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>2.5</td>
      <td>1510577970</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>1</td>
      <td>4.5</td>
      <td>1305696483</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_movie_rating = movie_data.pivot_table(index='userId', columns='title', values='rating')
```


```python
user_movie_rating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9719 columns</p>
</div>



Now lets do the case study of movie "eXistenZ (1999)" and recommend similar movies to the users


```python
ex_ratings = user_movie_rating['eXistenZ (1999)']
ex_ratings
```




    userId
    1      NaN
    2      NaN
    3      NaN
    4      NaN
    5      NaN
          ... 
    606    NaN
    607    NaN
    608    4.5
    609    NaN
    610    NaN
    Name: eXistenZ (1999), Length: 610, dtype: float64




```python
movies_like_ex = user_movie_rating.corrwith(ex_ratings)

corr_ex = pd.DataFrame(movies_like_ex, columns=['Correlation'])
corr_ex.dropna(inplace=True)
corr_ex.head()
```

    C:\Users\Shreyas\anaconda3\lib\site-packages\numpy\lib\function_base.py:2526: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\Shreyas\anaconda3\lib\site-packages\numpy\lib\function_base.py:2455: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'burbs, The (1989)</th>
      <td>-0.067522</td>
    </tr>
    <tr>
      <th>(500) Days of Summer (2009)</th>
      <td>-0.388883</td>
    </tr>
    <tr>
      <th>*batteries not included (1987)</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10 Things I Hate About You (1999)</th>
      <td>-0.396883</td>
    </tr>
    <tr>
      <th>10,000 BC (2008)</th>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_ex.sort_values('Correlation', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ides of March, The (2011)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Norma Rae (1979)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>That Obscure Object of Desire (Cet obscur objet du désir) (1977)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Excess Baggage (1997)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Executive Decision (1996)</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



From the output you can see that the movies that have high correlation with "eXistenZ (1999)" are not very well known. This shows that correlation alone is not a good metric for similarity because there can be a user who watched '"eXistenZ (1999)" and only one other movie and rated both of them as 5


```python
corr_ex = corr_ex.join(ratings_mean_count['rating_counts'])
corr_ex.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'burbs, The (1989)</th>
      <td>-0.067522</td>
      <td>17</td>
    </tr>
    <tr>
      <th>(500) Days of Summer (2009)</th>
      <td>-0.388883</td>
      <td>42</td>
    </tr>
    <tr>
      <th>*batteries not included (1987)</th>
      <td>1.000000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10 Things I Hate About You (1999)</th>
      <td>-0.396883</td>
      <td>54</td>
    </tr>
    <tr>
      <th>10,000 BC (2008)</th>
      <td>0.500000</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



You can see that the movie "*batteries not included (1987)", which has the highest correlation has only 7 ratings. This means that only 7 users gave same ratings to "eXistenZ (1999)", "*batteries not included (1987)". However, we can deduce that a movie cannot be declared similar to the another movie based on just 7 ratings. 


```python
corr_ex[corr_ex ['rating_counts']>50].sort_values('Correlation', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story 3 (2010)</th>
      <td>1.000000</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Avengers, The (2012)</th>
      <td>1.000000</td>
      <td>69</td>
    </tr>
    <tr>
      <th>Guardians of the Galaxy (2014)</th>
      <td>1.000000</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Ratatouille (2007)</th>
      <td>1.000000</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Zombieland (2009)</th>
      <td>0.996116</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>



We can now see that the movie "eXistenZ (1999)" is more relatable to movies like "Toy Story 3 (2010)" and "Avengers (2012)" as both of them recieve the similar ratings by common users and the number of users is also greater than 50 denoting a large scale opinion free from any kind of biases.

According to Wikipedia,the plot of the movie "eXistenZ (1999)" is "In the near-future, biotechnological virtual reality game consoles known as "game pods" have replaced electronic ones. The pods present "UmbyCords" that attach to "bio-ports", connectors surgically inserted into players' spines. Two game companies, Antenna Research and Cortical Systematics, compete against each other. In addition, the Realists fight both companies to prevent the "deforming" of reality". this plot is somewhat similar to "Toy story 3" and "Avengers". this proves thst our recommender system performed well on the given dataset


```python

```
