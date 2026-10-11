"""Server entry point: starts loading the agent as soon as the server boots, so the models and
indexes are usually ready before anyone opens the page.

Run:  streamlit run server.py      (the UI itself lives in app.py)
"""
from contextlib import asynccontextmanager

import streamlit as st


@asynccontextmanager
async def lifespan(app):
    from resources import start_loading   # import after the Streamlit runtime has started

    start_loading()   # background thread; the page shows a progress bar until it finishes
    yield


app = st.App("app.py", lifespan=lifespan)
