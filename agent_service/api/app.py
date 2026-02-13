#frontend

import streamlit as st
from PIL import Image
import requests
import json
import re
from pathlib import Path
import pandas as pd



BASE_URL = "http://127.0.0.1:8000"
CIT_PARSE_RE = re.compile(r"DOC:(?P<doc_id>[A-Za-z0-9_\-]+)\.c(?P<chunk_id>\d+)")

def parse_citation(citation: str):
    match = CIT_PARSE_RE.search(citation)
    if not match:
        return None
    
    return {
        "doc_id": match.group("doc_id"),
        "chunk_id": int(match.group("chunk_id"))
    }
st.set_page_config(
    page_title = "Climate AI Agent",
    page_icon = ":bar_chart",
    layout = "wide"
)

with st.sidebar:
    st.header("Filters", divider = True)
    selected_doc_type = st.multiselect("Document Type: ", ["Any", "methodology", "dictionary", "standard", "report", "presentation", "qa"])
    selected_juris = st.multiselect("Jurisdiction: ", ["Any", "nz", "au", "uk", "eu", "us", "apac", "global"])
    selected_peril = st.multiselect("Peril: ", ["Any", "flood", "heat", "fire", "wind", "slr", "drought"])
    selected_cluster_label_contains = st.text_input("Cluster label contains: ")
    top_k = st.number_input("Top K (default 8): ", min_value=1, max_value=50, value=8, step=1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            for source in message.get("sources").keys():
                with st.expander(f"{source}", expanded=False):
                    st.markdown(f"{message.get("sources")[source]}")
        

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = {"question": str(prompt), "doc_type": (selected_doc_type), "jurisdiction": (selected_juris), "peril": (selected_peril), "cluster_contains": str(selected_cluster_label_contains), "top_k": int(top_k)}
    with st.spinner(text="Thinking...", show_time=False, width="content"):
        response = requests.post(url=f"{BASE_URL}/ask",
                                data=json.dumps(inputs),
                                headers={"Content-Type": "application/json"})
    if response.status_code != 200:
        st.error(f"API error {response.status_code}")
        st.code(response.text)   # shows FastAPI validation details
        st.stop()
    # Display assistant response in chat message container
    #with st.chat_message("assistant"):
    #    st.markdown(response)
    # Add assistant response to chat history
    #st.session_state.messages.append({"role": "assistant", "content": response})
    if response.status_code == 200:
        result = response.json().get("answer")
        citations_used = response.json().get("citations")
        status = response.json().get("status")
        if status == "ok" or status == "fixed_citations":
            expander_content = {}
            #st.success(f"{status}")
            with st.chat_message("assistant"):
                st.markdown(result)

                if citations_used:
                    for citation in citations_used:
                        parsed_cit = parse_citation(citation)
                        chunks_file = Path(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library\03_chunks\{}.chunks.jsonl".format(parsed_cit["doc_id"]))
                        chunk_df = pd.read_json(path_or_buf = chunks_file, lines = True)
                        chunk_text = chunk_df.loc[chunk_df["chunk_id"] == parsed_cit["chunk_id"], "text"].iloc[0]
                        expander_content[f"{parsed_cit["doc_id"]} chunk: {parsed_cit["chunk_id"]}"] = chunk_text
                        citation_expander = st.expander(f"{parsed_cit["doc_id"]} chunk: {parsed_cit["chunk_id"]}")
                        with citation_expander:
                            st.write(f"{chunk_text}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result,
                    "sources": expander_content
                })
                #st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            st.error(f"{status}")
       
        
                        
    else:
        st.error("Error connecting to the API")
