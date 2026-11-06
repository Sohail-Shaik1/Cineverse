
import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineVerse | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. API CONFIGURATION
# =========================================================

try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    API_KEY = os.environ.get("TMDB_API_KEY", "")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/"


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, #252044 0%, transparent 35%),
        linear-gradient(135deg, #09090f, #11111c 60%, #09090f);
    color: white;
}

[data-testid="stSidebar"] {
    background: #10101b;
    border-right: 1px solid #29293a;
}

.hero {
    padding: 48px 35px;
    border-radius: 22px;
    background:
        linear-gradient(90deg, rgba(8,8,15,0.97), rgba(8,8,15,0.65)),
        linear-gradient(135deg, #6328a8, #15152a);
    border: 1px solid #37304e;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero p {
    color: #c7c4d6;
    font-size: 17px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 18px;
    color: white;
}

.movie-title {
    font-weight: 700;
    font-size: 15px;
    color: white;
    margin-top: 8px;
}

.movie-info {
    color: #aaa8bb;
    font-size: 12px;
}

div.stButton > button {
    background: linear-gradient(90deg, #8138ec, #b13cf2);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

div.stButton > button:hover {
    border: 1px solid #d9a7ff;
    color: white;
    background: linear-gradient(90deg, #9b4cff, #c64cff);
}

input, textarea {
    border-radius: 10px !important;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. DATASET LOCATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "movie_list (1).pkl"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "movie_list.pkl"


# =========================================================
# 5. LOAD DATASET AND BUILD RECOMMENDATION MODEL
# =========================================================

@st.cache_resource
def load_project():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Movie dataset not found. Upload "
            "'movie_list (1).pkl' to your GitHub repository."
        )

    with open(DATA_PATH, "rb") as file:
        movie_data = pickle.load(file)

    if not isinstance(movie_data, pd.DataFrame):
        raise ValueError(
            "The movie dataset is not a pandas DataFrame."
        )

    required_columns = ["movie_id", "title_x", "tags"]

    for column in required_columns:
        if column not in movie_data.columns:
            raise ValueError(
                f"Dataset is missing the required column: {column}"
            )

    movie_data = movie_data.copy().reset_index(drop=True)

    movie_data["tags"] = (
        movie_data["tags"].fillna("").astype(str)
    )

    movie_data["title_x"] = (
        movie_data["title_x"].fillna("").astype(str)
    )

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        movie_data["tags"]
    )

    return movie_data, vectors


# =========================================================
# 6. RECOMMENDATION FUNCTION
# =========================================================

def recommend(movie_title, movie_data, vectors, number=10):

    matches = movie_data[
        movie_data["title_x"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movie_indices = scores.argsort()[::-1]

    movie_indices = [
        index
        for index in movie_indices
        if index != movie_index
    ][:number]

    return movie_data.iloc[movie_indices]


# =========================================================
# 7. TMDB API FUNCTIONS
# =========================================================

def tmdb_request(endpoint, params=None):

    if not API_KEY:
        return None

    request_params = params.copy() if params else {}
    request_params["api_key"] = API_KEY

    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=request_params,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


@st.cache_data(ttl=21600)
def get_movie_details(movie_id):

    return tmdb_request(
        f"movie/{int(movie_id)}"
    )


@st.cache_data(ttl=21600)
def get_trending_movies():

    data = tmdb_request(
        "trending/movie/week",
        {"language": "en-US"}
    )

    if data:
        return data.get("results", [])

    return []


@st.cache_data(ttl=21600)
def search_tmdb_movies(query):

    data = tmdb_request(
        "search/movie",
        {
            "query": query,
            "language": "en-US",
            "include_adult": False
        }
    )

    if data:
        return data.get("results", [])

    return []


def poster_url(path, size="w500"):

    if not path:
        return None

    return f"{IMAGE_URL}{size}{path}"


# =========================================================
# 8. MOVIE CARD
# =========================================================

def movie_card(movie, show_details=True):

    movie_id = movie.get("id")

    if not movie_id:
        return

    title = movie.get("title", "Unknown Movie")
    poster = poster_url(movie.get("poster_path"))
    rating = movie.get("vote_average") or 0
    release_date = movie.get("release_date", "")

    with st.container():

        if poster:
            st.image(poster, width="stretch")
        else:
            st.markdown("### 🎬 Poster unavailable")

        st.markdown(
            f"<div class='movie-title'>{title}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>⭐ {rating:.1f}/10</div>",
            unsafe_allow_html=True
        )

        if release_date:
            st.caption(f"Release: {release_date}")

        if show_details:

            if st.button(
                "View Details",
                key=f"details_{movie_id}",
                use_container_width=True
            ):
                st.session_state["selected_movie_id"] = movie_id


# =========================================================
# 9. MOVIE DETAILS
# =========================================================

def show_movie_details(movie_id):

    details = get_movie_details(movie_id)

    if not details:
        st.error("Unable to load movie details.")
        return

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:

        poster = poster_url(
            details.get("poster_path")
        )

        if poster:
            st.image(poster, width="stretch")

    with col2:

        st.title(
            details.get("title", "Movie")
        )

        st.write(
            details.get(
                "overview",
                "No description available."
            )
        )

        rating = details.get("vote_average") or 0

        st.metric(
            "⭐ TMDB Rating",
            f"{rating:.1f}/10"
        )

        st.write(
            "**Release Date:**",
            details.get("release_date", "Not available")
        )

        st.write(
            "**Runtime:**",
            f"{details.get('runtime', 'N/A')} minutes"
        )

        genres = [
            genre["name"]
            for genre in details.get("genres", [])
        ]

        st.write(
            "**Genres:**",
            ", ".join(genres) if genres else "Not available"
        )

        st.write(
            "**Status:**",
            details.get("status", "Not available")
        )

        st.link_button(
            "Open on TMDB ↗",
            f"https://www.themoviedb.org/movie/{movie_id}"
        )


# =========================================================
# 10. LOAD PROJECT
# =========================================================

try:

    movies, vectors = load_project()

except Exception as error:

    st.error(f"Project loading error: {error}")

    st.info(
        "Check that movie_list (1).pkl is uploaded "
        "to the same GitHub folder as this application."
    )

    st.stop()


# =========================================================
# 11. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# 🎬 CINEVERSE")

    st.caption("YOUR PERSONAL MOVIE UNIVERSE")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔍 Discover Movies",
            "ℹ️ About"
        ],
        key="navigation"
    )

    st.markdown("---")

    st.markdown("### 🎥 Your Movie World")

    st.caption(
        f"Movies in database: {len(movies):,}"
    )

    st.markdown("---")

    st.caption("Powered by TMDB")


# =========================================================
# 12. HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to CineVerse 🍿</h1>
        <p>
            Discover movies. Explore new worlds.
            Find your next favourite film.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not API_KEY:

        st.warning(
            "TMDB API key is missing. Add TMDB_API_KEY "
            "in your Streamlit Cloud Secrets."
        )

    st.markdown(
        "<div class='section-title'>🔥 Trending This Week</div>",
        unsafe_allow_html=True
    )

    trending = get_trending_movies()

    if trending:

        columns = st.columns(5)

        for index, movie in enumerate(trending[:10]):

            with columns[index % 5]:
                movie_card(movie)

    else:

        st.info(
            "Trending movies are currently unavailable."
        )

    st.markdown(
        "<div class='section-title'>✨ Find Your Next Favourite</div>",
        unsafe_allow_html=True
    )

    movie_options = sorted(
        movies["title_x"].unique().tolist()
    )

    selected_movie = st.selectbox(
        "Choose a movie you already love",
        movie_options
    )

    if st.button(
        "🚀 Recommend Movies For Me",
        use_container_width=True
    ):

        with st.spinner(
            "Finding movies you might love..."
        ):

            recommendations = recommend(
                selected_movie,
                movies,
                vectors,
                number=10
            )

            st.session_state["home_recommendations"] = (
                recommendations
            )

            st.session_state["home_selected_movie"] = (
                selected_movie
            )

    if "home_recommendations" in st.session_state:

        st.success(
            "Movies similar to "
            + st.session_state["home_selected_movie"]
        )

        recommendations = st.session_state[
            "home_recommendations"
        ]

        columns = st.columns(5)

        for index, (_, row) in enumerate(
            recommendations.iterrows()
        ):

            details = get_movie_details(
                row["movie_id"]
            )

            with columns[index % 5]:

                if details:
                    movie_card(
                        details,
                        show_details=False
                    )
                else:
                    st.markdown(
                        f"**{row['title_x']}**"
                    )


# =========================================================
# 13. DISCOVER MOVIES PAGE
# =========================================================

elif page == "🔍 Discover Movies":

    st.markdown(
        "<div class='section-title'>🔍 Discover Movies</div>",
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Search for your favourite movie",
        placeholder="Example: Interstellar, Avatar, Inception..."
    )

    if query:

        with st.spinner("Searching movies..."):

            results = search_tmdb_movies(query)

        if results:

            columns = st.columns(5)

            for index, movie in enumerate(results[:15]):

                with columns[index % 5]:
                    movie_card(movie)

        else:

            st.info(
                "No movies found. Try another search."
            )

    st.markdown("---")

    st.markdown(
        "<div class='section-title'>🎞️ Explore Our Collection</div>",
        unsafe_allow_html=True
    )

    collection_search = st.text_input(
        "Search in our recommendation dataset",
        placeholder="Enter a movie title..."
    )

    if collection_search:

        filtered = movies[
            movies["title_x"].str.contains(
                collection_search,
                case=False,
                na=False
            )
        ]

        if filtered.empty:

            st.warning(
                "Movie not found in our collection."
            )

        else:

            for _, row in filtered.head(10).iterrows():

                col1, col2 = st.columns([3, 1])

                with col1:

                    st.write(
                        f"🎬 **{row['title_x']}**"
                    )

                with col2:

                    if st.button(
                        "Recommend",
                        key=f"discover_{row['movie_id']}"
                    ):

                        recommendations = recommend(
                            row["title_x"],
                            movies,
                            vectors,
                            number=10
                        )

                        st.session_state[
                            "discover_recommendations"
                        ] = recommendations

                        st.session_state[
                            "discover_title"
                        ] = row["title_x"]

    if "discover_recommendations" in st.session_state:

        st.markdown("---")

        st.subheader(
            "Recommendations for "
            + st.session_state["discover_title"]
        )

        recommendations = st.session_state[
            "discover_recommendations"
        ]

        columns = st.columns(5)

        for index, (_, row) in enumerate(
            recommendations.iterrows()
        ):

            details = get_movie_details(
                row["movie_id"]
            )

            with columns[index % 5]:

                if details:

                    movie_card(
                        details,
                        show_details=False
                    )

                else:

                    st.write(
                        row["title_x"]
                    )


# =========================================================
# 14. SELECTED MOVIE DETAILS
# =========================================================

if (
    page != "ℹ️ About"
    and "selected_movie_id" in st.session_state
):

    show_movie_details(
        st.session_state["selected_movie_id"]
    )


# =========================================================
# 15. ABOUT PAGE
# =========================================================

if page == "ℹ️ About":

    st.markdown("""
    <div class="hero">
        <h1>About CineVerse 🎬</h1>
        <p>
            Your personal movie discovery and
            recommendation platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("What is CineVerse?")

    st.write("""
    CineVerse is a movie recommendation system built using
    Python, Machine Learning, Streamlit and The Movie Database
    (TMDB) API.

    It helps users discover movies based on their interests
    and find similar movies using content-based filtering.
    """)

    st.subheader("✨ Features")

    st.markdown("""
    - 🎬 Movie recommendations
    - 🔥 Trending movies
    - 🔍 Movie search
    - 🖼️ Movie posters
    - ⭐ Ratings and movie information
    - 🤖 Content-based filtering
    - 🌙 Premium dark interface
    """)

    st.subheader("🛠️ Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TMDB API
    - CountVectorizer
    - Cosine Similarity
    """)


# =========================================================
# 16. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#aaa; padding:15px;">

    <h3>🎬 CINEVERSE</h3>

    <p>Discover. Explore. Experience Movies.</p>

    <p style="font-size:12px;">
        This product uses the TMDB API but is not endorsed
        or certified by TMDB.
    </p>

    <p style="font-size:12px;">
        Made with ❤️ using Python and Streamlit
    </p>

</div>
""", unsafe_allow_html=True)
