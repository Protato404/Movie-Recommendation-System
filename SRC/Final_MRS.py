import pandas as pd
import customtkinter as ctk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tkinter import messagebox
import os


# LOAD DATASET
movies = pd.read_csv("MRS_dataset/movies.csv", low_memory=False)
tags = pd.read_csv("MRS_dataset/tags.csv", low_memory=False)
links = pd.read_csv("MRS_dataset/links.csv", low_memory=False)

movies = movies.head(len(movies) // 4)
valid_ids = movies["movieId"]

tags = tags[tags["movieId"].isin(valid_ids)]
links = links[links["movieId"].isin(valid_ids)]


tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str)))
movies = movies.merge(tags_grouped, on="movieId", how="left")
movies = movies.merge(links, on="movieId")
movies["tag"] = movies["tag"].fillna("")

# Combine features
movies["content"] = (movies["genres"] + " " + movies["tag"])

# VECTORIZE
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies["content"]).toarray()
similarity = cosine_similarity(vectors)

#
def find_poster(tmdb_id):

    poster_path = f"MRS_posters/{tmdb_id}"

    if os.path.exists(poster_path):
        image = Image.open(poster_path)
        image = image.resize((150, 220))

        return ctk.CTkImage(light_image=image, dark_image=image, size=(150, 220))

    return None

# RECOMMEND FUNCTION
def recommend(movie_name):

    for widget in result_frame.winfo_children():
        widget.destroy()



    if movie_name.isdigit() or movie_name.strip() == "":
        messagebox.showerror("Invalid Input!", "Please Enter a Valid title")
        return

    matches = movies[movies["title"].str.lower().str.contains(movie_name.lower(), regex=False)]
    if matches.empty:
        messagebox.showerror("Movie Not found!", f"{movie_name} was not in the dataset...")
        return

    movie_index = matches.index[0]
    distances = similarity[movie_index]

    recommended = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1])[1:7]


    selected_title = movies.iloc[movie_index]["title"]
    selected_tmdb = movies.iloc[movie_index]["tmdbId"]

    movie_frame = ctk.CTkFrame(result_frame, fg_color="gray", corner_radius=15)
    movie_frame.grid(row=0, column=0, padx= 15, pady= 15)
    poster = find_poster(selected_tmdb)


    if poster:
        btn = ctk.CTkButton(movie_frame, image= poster, text= "", width= 150, height= 220, fg_color= "transparent")
        btn.pack(pady = 10)
    movie_label = ctk.CTkLabel(movie_frame, text= selected_title, wraplength= 140) #font = ("Arial", 14, "bold")
    movie_label.pack(pady = 10)

    row_num = 0
    column_num = 1
    for movie in recommended:
        index = movie[0]
        title = movies.iloc[index].title
        tmdb_id = movies.iloc[index].tmdbId
        if title.lower() == selected_title.lower():
            continue

        movie_frame = ctk.CTkFrame(result_frame, fg_color="black", corner_radius=15)
        movie_frame.grid(row=row_num, column=column_num, padx=15, pady=15)
        column_num += 1
        if column_num > 4:
            column_num = 0
            row_num += 1

        poster = find_poster(tmdb_id)
        if poster:
            btn = ctk.CTkButton(movie_frame, image=poster, text="", width=150, height=220, fg_color="transparent", hover=False, command=lambda t=title: recommend(t))
            btn.pack(pady=10)


        movie_label = ctk.CTkLabel(movie_frame, text=title, wraplength=140, font=("Arial", 14, "bold"))
        movie_label.pack(pady=10)

def show_home_movies():
    for widget in result_frame.winfo_children():
        widget.destroy()

    sample_movies = movies.head(20) #test 5000

    row_num = 0
    col_num = 0

    for _, row in sample_movies.iterrows():

        title = row.title
        tmdb_id = row.tmdbId

        movie_frame = ctk.CTkFrame(result_frame, fg_color="black", corner_radius=15)
        movie_frame.grid(row=row_num, column=col_num, padx=15, pady=15)

        poster = find_poster(tmdb_id)

        if poster:
            btn = ctk.CTkButton(movie_frame, image=poster, text="", width=150, height=220, fg_color="transparent", hover=False, command=lambda t=title: recommend(t))
            btn.pack(pady=10)

        movie_label = ctk.CTkLabel(movie_frame, text=title, wraplength=140, font=("Arial", 14, "bold"))
        movie_label.pack(pady=10)
        col_num += 1

        if col_num > 4:
            col_num = 0
            row_num += 1

#========================================
# GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()
app.geometry("1050x700")
app.title("Movie System Recommender")
app.resizable(False, False)

header = ctk.CTkFrame(app, height=80, fg_color="black")
header.pack(fill="x")
title = ctk.CTkLabel(header, text="MOVIE RECOMMENDER", text_color="#E50914", font=("Arial", 34, "bold"))
title.pack(pady=20)

#Search bar
search_frame = ctk.CTkFrame(app, fg_color="transparent")
search_frame.pack(pady=20)
search_entry = ctk.CTkEntry(search_frame, width=500, height=50, placeholder_text="Search movie...", font=("Arial", 18))
search_entry.pack(side="left", padx=10)
recommend_btn = ctk.CTkButton(search_frame, text="Recommend", width=150, height=50, fg_color="#E50914", hover_color="#B20710", font=("Arial", 18, "bold"), command=lambda: recommend(search_entry.get()))
recommend_btn.pack(side="left")

#Show screen and results
result_frame = ctk.CTkScrollableFrame(app, width=1100, height=500, fg_color="#1A1A1A")
result_frame.pack(pady=20, padx=20, fill="both", expand=True)
show_home_movies()
app.mainloop()


print("Movies: ",movies.shape)
print("Tags: ",tags.shape)
print("links: ",links.shape)




