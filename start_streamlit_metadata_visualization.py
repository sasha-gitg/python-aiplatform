import streamlit as st

from google.cloud import aiplatform
from google.cloud.aiplatform.metadata import context

import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config


def hash_resource(resource):
	return hash(resource.gca_resource)


@st.cache(hash_funcs={context._Context: hash_resource})
def load_contexts(project=None, location='us-central1'):
	aiplatform.init(project=project, location=location)
	contexts = context._Context.list()
	return [c.to_dict() for c in contexts]

def str_context(context_dict):
	return f'{context_dict["schemaTitle"]}:{context_dict.get("displayName")}:{context_dict["name"].split("/")[-1]}'

contexts = load_contexts()
st.write(pd.DataFrame(contexts))

option = st.selectbox(
	'Select context',
	[str_context(d) for d in contexts]
)

 