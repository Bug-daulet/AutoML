import os
import streamlit as st
import pandas as pd


def app():
    trainDatasetFile = st.file_uploader("Upload your train dataset", type=['txt', 'csv', 'xlsx'],
                                        accept_multiple_files=False)
    testDatasetFile = st.file_uploader("Upload your test dataset", type=['txt', 'csv', 'xlsx'],
                                       accept_multiple_files=False)

    if trainDatasetFile is not None and testDatasetFile is not None:
        # train_file_details = {"FileName": trainDatasetFile.name, "FileType": trainDatasetFile.type}
        st.write("File name: ", trainDatasetFile.name)
        # test_file_details = {"FileName": testDatasetFile.name, "FileType": testDatasetFile.type}
        st.write("File name: ", testDatasetFile.name)

        if st.button("Save file"):
            with open(os.path.join("dataset", "train.xlsx"), "wb") as f:
                f.write(trainDatasetFile.getbuffer())

            with open(os.path.join("dataset", "test.xlsx"), "wb") as f:
                f.write(testDatasetFile.getbuffer())
            st.success("File Saved!")
