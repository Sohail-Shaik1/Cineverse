
import os
import pickle
import requests
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineVerse | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/"

# ---------------- PREMIUM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080b14;
    color: #ffffff;
}

header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: white !important;
    font-weight: 800 !important;
}

.hero {
    background: linear-gradient(100deg, #19132c, #351323, #111827);
    padding: 42px;
    border-radius: 22px;
    border: 1px solid #343047;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 48px !important;
    margin-bottom: 8px;
}

.hero p {
    color: #c6c4d1;
    font-size: 17px;
}

.movie-card {
    background: #151925;
    padding: 12px;
    border-radius: 14px;
    border: 1px solid #292e3d;
    min-height: 100px;
}

.movie-card h4 {
    font-size: 15px;
    color: white;
    margin-bottom: 7px;
}

.movie-card p {
    color: #b5b8c5;
    font-size: 12px;
}

div.stButton > button {
    background: linear-gradient(90deg, #e50914, #ff4b45);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
}

div.stButton > button:hover {
    background: #ff3039;
    color: white;
    border: none;
}

section[data-testid="stSidebar"] {
    background: #101522;
}

hr {
    border-color: #303647;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD PROJECT FILES ----------------
@st.cache_resource
def load_project():
    with open("/content/movie_list.pkl", "rb") as f:
        movie_data = pickle.load(f)

    with open("/content/similarity.pkl", "rb") as f:
        similarity_data = pickle.load(f)

    return movie_data, similarity_data


movies, similarity = load_project()


# ---------------- TMDB API ----------------
@st.cache_data(ttl=21600)
def get_movie_details(movie_id):
    try:
        response = requests.get(
            f"{BASE_URL}/movie/{movie_id}",
            params={"api_key": API_KEY},
            timeout=15
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return {}


@st.cache_data(ttl=21600)
def get_trending_movies():
    try:
        response = requests.get(
            f"{BASE_URL}/trending/movie/week",
            params={"api_key": API_KEY},
            timeout=15
        )

        response.raise_for_status()
        return response.json().get("results", [])

    except requests.RequestException:
        return []


# ---------------- RECOMMENDATION ENGINE ----------------
def recommend(movie_title):
    matched = movies[
        movies["title_x"].str.lower() == movie_title.lower()
    ]

    if matched.empty:
        return []

    index = matched.index[0]
    distances = similarity[index]

    movie_indices = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommendations = []

    for movie_index, score in movie_indices:
        row = movies.iloc[movie_index]

        recommendations.append({
            "title": row["title_x"],
            "movie_id": int(row["movie_id"]),
            "score": float(score)
        })

    return recommendations


# ---------------- MOVIE CARD ----------------
def display_movie_card(movie, show_score=False):
    movie_id = movie.get("id", movie.get("movie_id"))
    title = movie.get("title", "Unknown Movie")

    details = get_movie_details(movie_id)

    poster_path = details.get("poster_path")
    rating = details.get("vote_average", movie.get("vote_average", 0))
    release_date = details.get("release_date", "")

    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

    if poster_path:
        st.image(
            IMAGE_URL + "w500" + poster_path,
            use_container_width=True
        )
    else:
        st.markdown("🎬 Poster unavailable")

    st.markdown(f"**{title}**")
    st.caption(f"⭐ {rating:.1f}/10")

    if release_date:
        st.caption(release_date[:4])

    if show_score:
        st.caption(f"Match: {movie['score']:.1%}")

    st.markdown(
        f"[View movie ↗](https://www.themoviedb.org/movie/{movie_id})"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🎬 CineVerse")
    st.caption("Your personal movie universe")

    st.markdown("---")

    page = st.radio(
        "EXPLORE",
        ["Home", "Discover Movies", "About"]
    )

    st.markdown("---")
    st.caption("Powered by your movie recommendation engine")


# ---------------- HOME PAGE ----------------
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🎬 CINEVERSE</h1>
        <p>Discover stories. Explore worlds. Find your next favourite movie.</p>
        <p>Thousands of movies. Endless possibilities.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔥 Trending This Week")

    trending = get_trending_movies()

    if trending:
        cols = st.columns(5)

        for col, movie in zip(cols, trending[:5]):
            with col:
                display_movie_card(movie)
    else:
        st.info("Trending movies are temporarily unavailable.")

    st.markdown("---")

    st.subheader("🍿 Find Movies Similar to Your Favourite")

    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(movies["title_x"].dropna().unique().tolist())
    )

    if st.button("✨ Discover My Movies", use_container_width=True):

        with st.spinner("Finding movies you'll love..."):
            results = recommend(selected_movie)

        st.session_state["recommendations"] = results
        st.session_state["selected_movie"] = selected_movie

    if "recommendations" in st.session_state:

        st.markdown("## 🎯 Recommended For You")

        cols = st.columns(5)

        for col, movie in zip(
            cols,
            st.session_state["recommendations"][:5]
        ):
            with col:
                display_movie_card(movie, show_score=True)


# ---------------- DISCOVER PAGE ----------------
elif page == "Discover Movies":

    st.markdown("# 🔎 Discover Movies")
    st.caption("Search for a movie from our collection of 4,800 films.")

    search_movie = st.selectbox(
        "Search or select a movie",
        sorted(movies["title_x"].dropna().unique().tolist()),
        key="discover_movie"
    )

    if st.button("Explore Movie", use_container_width=True):

        matched = movies[
            movies["title_x"] == search_movie
        ]

        if not matched.empty:

            movie_id = int(matched.iloc[0]["movie_id"])

            with st.spinner("Loading movie information..."):
                details = get_movie_details(movie_id)

            st.markdown("---")

            left, right = st.columns([1, 2])

            with left:
                poster = details.get("poster_path")

                if poster:
                    st.image(
                        IMAGE_URL + "w500" + poster,
                        use_container_width=True
                    )
                else:
                    st.info("Poster unavailable")

            with right:
                st.title(details.get("title", search_movie))

                st.write(
                    f"⭐ {details.get('vote_average', 0):.1f}/10"
                )

                st.write(
                    f"📅 Release date: {details.get('release_date', 'Unknown')}"
                )

                st.write(
                    f"⏱️ Runtime: {details.get('runtime', 'Unknown')} minutes"
                )

                genres = details.get("genres", [])

                if genres:
                    st.write(
                        "🎭 Genres: " +
                        ", ".join(g["name"] for g in genres)
                    )

                st.write(
                    details.get("overview", "No description available.")
                )

                st.link_button(
                    "View on TMDB",
                    f"https://www.themoviedb.org/movie/{movie_id}"
                )

            st.markdown("---")

            st.subheader("🎯 Similar Movies")

            results = recommend(search_movie)
            cols = st.columns(5)

            for col, movie in zip(cols, results[:5]):
                with col:
                    display_movie_card(movie, show_score=True)


# ---------------- ABOUT PAGE ----------------
elif page == "About":

    st.markdown("# About CineVerse")

    st.write("""
    CineVerse is a content-based movie recommendation system.

    It uses movie information and cosine similarity to identify
    films with similar characteristics.

    **Project features**

    - Content-based movie recommendations
    - 4,800 movie records
    - TMDB movie posters
    - Real-time movie information
    - Ratings and descriptions
    - Trending movie discovery
    - Netflix-inspired dark interface
    """)

    st.markdown("---")
    st.write("Built with Python, Streamlit, Scikit-learn and TMDB.")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "Movie data and images provided by TMDB. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
