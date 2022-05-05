import streamlit as st
from multiapp import MultiApp
from apps import analyze, upload  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Upload", upload.app)
app.add_app("Train model", analyze.app)

# The main app
app.run()
