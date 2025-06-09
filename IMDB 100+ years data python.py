#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Information Technology\IMDB 100+ years data\imdb_100+_years_dataset.csv")
print(df.head())


# In[148]:


df.info()


# In[149]:


df.columns


# In[150]:


df.isnull().sum()


# In[151]:


# Drop highly missing or irrelevant columns
df.drop(columns=['Movie_Link', 'awards_content'], inplace=True)


# In[152]:


# Example fill for numerical columns
df['rating'] = df['rating'].fillna(df['rating'].median())
df['votes'] = df['votes'].fillna(0)


# In[153]:


# Example fill for categorical
df['MPA'] = df['MPA'].fillna("Not Rated")


# In[154]:


df.isnull().sum()


# In[158]:


# Drop rows with missing genres
df.dropna(subset=['genres'], inplace=True)


# In[163]:


import re

# Custom parser function
def parse_duration(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    
    # Match patterns like '2h 10m', '1h', '45m'
    match = re.match(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?', x)
    if not match:
        return np.nan

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes

# Apply conversion
df['duration'] = df['duration'].apply(parse_duration)


# In[165]:


df['duration'].describe()
df['duration'].isnull().sum()
df['duration'].dtype


# In[167]:


df['duration'].isnull().sum()


# In[168]:


df['duration'].describe()


# In[169]:


# Simple imputation
df['duration'] = df['duration'].fillna(df['duration'].median())


# In[175]:


df['duration'].isnull().sum()


# In[177]:


df.isnull().sum()


# In[179]:


df[df['meta_score'].notnull()]['meta_score'].describe()


# In[181]:


sns.histplot(df['meta_score'].dropna(), bins=20, color='purple')
plt.title("Distribution of Meta Scores")
plt.show()


# In[183]:


df['budget'].head()


# In[190]:


df.drop(columns=['budget'], inplace=True)


# In[192]:


df.columns


# In[198]:


df['gross_us_canada'].head(20)


# In[200]:


df['stars'].head(20) # objective column


# In[206]:


df['languages'].head(20) # again its an objective column, will analyze it letter to know about the language trend


# In[208]:


df['languages'].info() # checking the language column information


# In[210]:


df['countries_origin'].info() # It is an objective column


# In[212]:


df['countries_origin'].isnull().sum() # checking the null values in this column


# In[214]:


df.columns


# In[230]:


#  Filter movies released in 2024
df_2024 = df[df['year'] == 2024]

# Drop missing or empty genre entries
# df_2024 = df_2024[df_2024['genres'].notnull() & (df_2024['genres'] != '')]

#  Split genre strings (like 'Monster Horror, Sea Adventure') into separate entries
all_genres_2024 = df_2024['genres'].str.split(',').explode().str.strip()

#  Count top 5 genres
top_genres_2024 = all_genres_2024.value_counts().head(10)

# Display result
print("Top 10 genres in 2024:")
print(top_genres_2024)


# In[238]:


# Clean those square brackets and quotes before splitting

# Here’s how to fix it:

#  Filter for 2024 movies
df_2024 = df[df['year'] == 2024].copy()

# Remove square brackets and quotes from 'genres'
df_2024['genres'] = df_2024['genres'].str.replace(r"[\[\]\'\"]", '', regex=True)

#  Split the genres and explode into separate rows
all_genres_2024 = df_2024['genres'].str.split(',').explode().str.strip()

#  Get top 5 genres
top_genres_2024 = all_genres_2024.value_counts().head(10)

# Display result
print("Top 20 genres in 2024:")
print(top_genres_2024)


# In[246]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prepare top 10 genres
top_genres_2024 = all_genres_2024.value_counts().head(10).reset_index()
top_genres_2024.columns = ['genre', 'count']

# Plot using hue for color mapping
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_genres_2024,
    x='count',
    y='genre',
    hue='genre',  # assign hue to use palette
    palette='crest',
    dodge=False,  # no grouping
    legend=False  # disable legend if you don't need it
)

plt.title('Top 10 Movie Genres in 2024')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[252]:


# Clean those square brackets and quotes before splitting

# Here’s how to fix it:

# Filter for 2025 movies
df_2025 = df[df['year'] == 2025].copy()

# Remove square brackets and quotes from 'genres'
df_2025['genres'] = df_2025['genres'].str.replace(r"[\[\]\'\"]", '', regex=True)

#  Split the genres and explode into separate rows
all_genres_2025 = df_2025['genres'].str.split(',').explode().str.strip()

# Get top 5 genres
top_genres_2025 = all_genres_2025.value_counts().head(10)

#  Display result
print("Top 10 genres in 2025:")
print(top_genres_2025)


# In[254]:


# Prepare top 10 genres
top_genres_2025 = all_genres_2025.value_counts().head(10).reset_index()
top_genres_2025.columns = ['genre', 'count']

# Plot using hue for color mapping
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_genres_2025,
    x='count',
    y='genre',
    hue='genre',  # assign hue to use palette
    palette='crest',
    dodge=False,  # no grouping
    legend=False  # disable legend if you don't need it
)

plt.title('Top 10 Movie Genres in 2025')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[260]:


# Clean those square brackets and quotes before splitting

# Here’s how to fix it:

# Filter for 1995 movies
df_1995 = df[df['year'] == 1995].copy()

# Remove square brackets and quotes from 'genres'
df_1995['genres'] = df_1995['genres'].str.replace(r"[\[\]\'\"]", '', regex=True)

#  Split the genres and explode into separate rows
all_genres_1995 = df_1995['genres'].str.split(',').explode().str.strip()

# Get top 5 genres
top_genres_1995 = all_genres_1995.value_counts().head(10)

#  Display result
print("Top 10 genres in 1995:")
print(top_genres_1995)


# In[320]:


all_genres_1995.head() # checking after spliting the genres content


# In[262]:


# Prepare top 10 genres in 1995
top_genres_1995 = all_genres_1995.value_counts().head(10).reset_index()
top_genres_1995.columns = ['genre', 'count']

# Plot using hue for color mapping
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_genres_1995,
    x='count',
    y='genre',
    hue='genre',  # assign hue to use palette
    palette='crest',
    dodge=False,  # no grouping
    legend=False  # we can disable legend if we don't need it
)

plt.title('Top 10 Movie Genres in 1995')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[322]:


# print(df_1995.head())
df_1995['genres'].head(10)


# In[266]:


df.columns


# In[300]:


df['genres'].head(1)


# In[ ]:


df['description'].isnull().sum()


# In[272]:


df['rating'].head()


# In[290]:


# Sort by rating (and votes for tie-breaker)
top_20_movies = df.sort_values(by=['rating', 'votes'], ascending=[False, False]).head(20)

# Display selected columns
top_20_movies[['title', 'year', 'rating', 'votes']].reset_index(drop=True)


# In[292]:


plt.figure(figsize=(12, 8))
sns.barplot(x='rating', y='title', data=top_20_movies, palette='rocket', order=top_20_movies.sort_values('rating', ascending=False)['title'])
plt.title('Top 20 Highest Rated Movies of All Time')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()


# ### Calculating Most frequent genre of all time

# In[328]:


# Clean brackets and quotes
df['genres'] = df['genres'].str.replace(r"[\[\]\'\"]", '', regex=True)

#  Split genres and explode to individual rows
all_genres = df['genres'].str.split(',').explode().str.strip()

#  Count genres
genre_counts = all_genres.value_counts()

# Display the most common genre
print("Most frequent genre of all time:")
print(genre_counts.head())


# ###  Ploting top 10 movie genre in all time

# In[330]:


plt.figure(figsize=(10, 6)) # ploting top 10 movie genre in all time
sns.barplot(x=genre_counts.head(10).values, y=genre_counts.head(10).index, palette='viridis')
plt.title('Top 10 Most Common Movie Genres (All Time)')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[338]:


df.isnull().sum()


# ### ploting wordcloud to know the most frequent language of movie release in all time globaly

# In[344]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

#  Drop missing values in 'languages'
language_data = df['languages'].dropna()

# Clean brackets and quotes if stored as list-like strings
language_data = language_data.str.replace(r"[\[\]\'\"]", '', regex=True)

# Join all languages into one long string
language_text = ' '.join(language_data)

#  Generate the word cloud
wordcloud = WordCloud(width=1000, height=600, background_color='white', colormap='coolwarm').generate(language_text)

# Plot the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Languages in Movies')
plt.show()


# ### Ploting wordcloud to know most frequent stars in all time in IMDB

# In[347]:


#  Drop NaN values
stars_data = df['stars'].dropna()

#  Clean brackets and quotes if necessary
stars_data = stars_data.str.replace(r"[\[\]\'\"]", '', regex=True)

#  Split multiple stars per movie and join into one long string
all_stars = stars_data.str.split(',').explode().str.strip()
stars_text = ' '.join(all_stars)

#  Generate the word cloud
star_wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color='black',
    colormap='spring',
    contour_color='white',
    contour_width=1.5,
    max_words=200
).generate(stars_text)

# Plot it

plt.figure(figsize=(14, 8))
plt.imshow(star_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Movie Stars', fontsize=18, color='white')
plt.tight_layout()
plt.show()


# ### Most frequent stars in hindi movie

# In[356]:


# Filter only rows where 'languages' includes 'Hindi'
df_hindi = df[df['languages'].notna() & df['languages'].str.contains('Hindi', case=False)]

# Drop missing stars
stars_hindi = df_hindi['stars'].dropna()

# Clean the strings (remove brackets, quotes)
stars_hindi = stars_hindi.str.replace(r"[\[\]\'\"]", '', regex=True)

# Split and explode the stars column
all_stars_hindi = stars_hindi.str.split(',').explode().str.strip()

# Join into one big string
stars_text_hindi = ' '.join(all_stars_hindi)

# Generate the word cloud
wordcloud_hindi_stars = WordCloud(
    width=1200,
    height=700,
    background_color='white',
    colormap='autumn',
    contour_color='black',
    contour_width=2,
    max_words=200
).generate(stars_text_hindi)

# Plot
plt.figure(figsize=(14, 8))
plt.imshow(wordcloud_hindi_stars, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Stars in Hindi Movies', fontsize=18)
plt.tight_layout()
plt.show()


# ### Top 10 stars who worked in hindi movies

# In[361]:


# Filter only Hindi language movies
df_hindi = df[df['languages'].notna() & df['languages'].str.contains('Hindi', case=False)]

# Drop missing 'stars' values
stars_hindi = df_hindi['stars'].dropna()

# Clean brackets/quotes
stars_hindi = stars_hindi.str.replace(r"[\[\]\'\"]", '', regex=True)

# Split multiple stars and explode
all_stars = stars_hindi.str.split(',').explode().str.strip()

# Count appearances
top_indian_actors = all_stars.value_counts().head(10)

# Display results
print("Top 10 Highest Working Indian Actors (based on appearance in Hindi movies):")
print(top_indian_actors)


# In[363]:


plt.figure(figsize=(10,6))
sns.barplot(x=top_indian_actors.values, y=top_indian_actors.index, palette='Set2')
plt.title('Top 10 Highest Working Indian Actors in IMDb Hindi Movies')
plt.xlabel('Number of Movie Appearances')
plt.ylabel('Actor')
plt.tight_layout()
plt.show()


# ### Lets find all the movies where Amitabh Bachchan is listed as a star and view their ratings

# In[377]:


# Filter rows where 'stars' is not null and includes 'Amitabh Bachchan'
amitabh_movies = df[df['stars'].notna() & df['stars'].str.contains('Amitabh Bachchan', case=False, na=False)]

# Select relevant columns: title, year, rating
amitabh_movie_ratings = amitabh_movies[['title', 'year', 'rating']].sort_values(by='rating', ascending=False)

# Display the result
print("Amitabh Bachchan Movie Ratings:")
print(amitabh_movie_ratings)


# ## IMDB reatings for only amitabh bachchan movies

# In[385]:


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(amitabh_movie_ratings['rating'], amitabh_movie_ratings['year'], marker='o', linestyle='-', color='darkblue')

# Titles and labels
plt.title("IMDb Ratings of Amitabh Bachchan Movies (Descending by Year)", fontsize=16)
plt.xlabel("IMDb Rating", fontsize=12)
plt.ylabel("Year", fontsize=12)
plt.gca().invert_xaxis()  # Optional: Descending year order left to right
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Here is all the movies of amitabh bachchan which reating fall between 6 to 10

# In[397]:


# Filter all Amitabh Bachchan movies with non-null ratings
amitabh_movies = df[df['stars'].notna() & df['stars'].str.contains('Amitabh Bachchan', case=False, na=False)]
amitabh_movies = amitabh_movies[amitabh_movies['rating'].notna()]

# Count total rated movies
total_movies = len(amitabh_movies)

# Filter movies with ratings between 6 and 10
movies_6_to_10 = amitabh_movies[(amitabh_movies['rating'] >= 6) & (amitabh_movies['rating'] <= 10)]
count_6_to_10 = len(movies_6_to_10)

# Show result
print(f"Total Amitabh Bachchan movies with ratings: {total_movies}")
print(f"Movies rated between 6 and 10: {count_6_to_10}")
print(f"Percentage: {count_6_to_10 / total_movies * 100:.2f}%")


# In[402]:


df.columns


# ## Wordcloud for country origin of films

# In[406]:


# Drop missing values in 'countries origin'
countries_origin_data = df['countries_origin'].dropna()

# Clean brackets and quotes if stored as list-like strings
countries_origin_data = countries_origin_data.str.replace(r"[\[\]\'\"]", '', regex=True)

# Join all languages into one long string
countries_origin_text = ' '.join(countries_origin_data)

# Generate the word cloud
wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='coolwarm').generate(countries_origin_text)

# Plot the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common countries origin of Movies')
plt.show()


# In[411]:


# Drop missing values in 'directors'
directors_data = df['directors'].dropna()

# Clean brackets and quotes if stored as list-like strings
directors_data = directors_data.str.replace(r"[\[\]\'\"]", '', regex=True)

# Join all directors into one long string
directors_text = ' '.join(directors_data)

# Generate the word cloud
wordcloud = WordCloud(width=1000, height=600, background_color='green', colormap='coolwarm').generate(directors_text)

# Plot the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common directors of all time')
plt.show()


# In[433]:


# Drop rows with missing directors
df_directors = df[df['directors'].notna()]

# Split multiple directors per movie, explode them into separate rows
all_directors = df_directors['directors'].str.split(',').explode().str.strip()

# Count occurrences and get top 10
top_10_directors = all_directors.value_counts().head(10)

# Display result
print("Top 10 Directors by Number of Movies:")
print(top_10_directors)


# # What we have seen so far is the EDA of IMDB movies. So we can analyze many more things as per the business need. THANK YOU!

# In[ ]:




